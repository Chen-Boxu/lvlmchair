import gradio as gr

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
SHARE = False
import shutil
import argparse
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn
import numpy as np
import pandas as pd
import gc

import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F

import cv2
from models import *

from sklearn.decomposition import PCA

# ../llava-1.5-7b-hf accelerate==0.21.0 transformers==4.41.2 tokenizers==0.15.2
# ./llava-1.5-13b-hf accelerate==0.26.0 transformers-4.47.1 tokenizers==0.21.0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="../llava-1.5-7b-hf")  # llava-1.5-7b-hf, minigpt4-7b, blip2-opt-6.7b
    parser.add_argument("--cfg-path", type=str, default='./minigpt4/minigpt4_eval.yaml')
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


# global model
print('Initializing Chat')
args = parse_args()
if 'llava-v1.5' in args.model_name:
    sys = LlavaForAnalysis(args)
elif 'llava-1.5' in args.model_name:
    sys = LlavaHFForAnalysis(args)
elif 'blip2' in args.model_name:
    sys = Blip2ForAnalysis(args)
elif 'minigpt4' in args.model_name:
    sys = MiniGPT4ForAnalysis(args)
else:
    raise NotImplementedError


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# def gradio_reset():
#     flush()
#     # [chatbot, text_input, upload_button, ans_tokens_to_select, layer_select, result_img, result_qus, result_ans, printstr, vc_img, max_vc], 
#     return gr.update(value=None), gr.update(placeholder='Please upload your image first', interactive=False), \
#            gr.update(value="Upload Picture", interactive=False), gr.update(choices=[],value=None), gr.update(value=32), \
#            None, None, None, None, None, None, None, None, None, \
#            gr.update(value=1)
# def gradio_reset():
#     flush()
#     return gr.update(value=None), gr.update(placeholder='Please upload your image first', interactive=False), \
#            gr.update(value="Upload Picture", interactive=False), gr.update(choices=[],value=None), gr.update(value=32), \
#            None, None, None, None, None, None, None, None, None, \
#            gr.update(value=1)
def gradio_reset():
    flush()
    # [chatbot, text_input, upload_button, ans_tokens_to_select, layer_select, result_img, result_qus, result_ans, printstr, vc_img, max_vc], 
    return gr.update(value=None), gr.update(placeholder='Please upload your image first', interactive=False)


def upload_img(gr_img):
    if gr_img is None:
        return None, None, gr.update(interactive=True)

    return gr_img, gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False)


# def upload_img(raw_img):
#     if raw_img is None:
#         return None, None, gr.update(interactive=True)

#     return raw_img, raw_img, gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False)


def gradio_ask(chatbot, text_input):
    if len(text_input) == 0:
        return chatbot, gr.update(interactive=True, placeholder='Input should not be empty!')
    flush()
    chatbot = chatbot + [[text_input, None]]

    return chatbot, None, None, None, None, None, None

def gradio_answer(chatbot, gr_img, question, temperature, num_beams):

    qus_tokens = sys.tokenizer.tokenize(question)
    
    # first generation forward pass
    sys.args.temperature = temperature
    sys.args.num_beams = num_beams
    ans_ids = sys.chat(gr_img, question)
    answer = sys.tokenizer.decode(ans_ids, skip_special_tokens=False)

    # second parrallel forward pass
    sys.register_hooks()
    outputs = sys.forward_with_grads(gr_img, question, answer)
    sys.remove_hooks()

    # per-token backward to compute relevancy scores
    relevancy_scores, probs = [], []
    for word_idx, logits in enumerate(outputs.logits):
        logits = logits.unsqueeze(0)
        token_id_one_hot = F.one_hot(ans_ids[word_idx], num_classes=logits.size(-1)).float().to(logits.device)
        token_id_one_hot = token_id_one_hot.view(1, -1)
        token_id_one_hot.requires_grad_(True)
        sys.model.zero_grad()
        logits.backward(gradient=token_id_one_hot, retain_graph=True)
        R = sys.compute_relevancy(word_idx)
        relevancy_scores.append(R)
        probs.append(torch.softmax(logits, dim=1).detach().max().item())
    
    chatbot[-1][1] = answer
    ans_tokens = [sys.tokenizer.decode(id) for id in ans_ids]
    ans_tokens_to_select = [f'{index}:{token}' for index, token in enumerate(ans_tokens)]

    # linear probe to obtain per-layer prediction evolution (greedy search)
    ans_hidden_states = torch.stack(outputs.hidden_states)
    lm_head = sys.model.lm_head
    with torch.no_grad():
        num_layers, ans_len, _ = ans_hidden_states.shape
        ans_hidden_states = ans_hidden_states.flatten(0, 1)
        ans_logits = lm_head(ans_hidden_states.to(lm_head.weight.device))
        ans_logits = ans_logits.view(num_layers, ans_len, -1)   # (1+32, ans_len, vocab_size)
    
    img_hidden_states = torch.stack(outputs.img_hidden_states)  # (1+32, n_img, dim) llava

    pca_tokens = []
    for tokens in img_hidden_states:
        pca = PCA(n_components=3)
        t = pca.fit_transform(tokens.detach().cpu().numpy())
        pca_tokens.append(torch.from_numpy(t))
    pca_tokens = torch.stack(pca_tokens)  # (1+32, n_img, 3)

    return chatbot, gr.update(choices=ans_tokens_to_select, interactive=True), relevancy_scores, probs, qus_tokens, ans_tokens, ans_logits, pca_tokens


def prob_plot(ans_tokens, probs):

    if len(ans_tokens) <= 32:
        fig = plt.figure(figsize=(15, 2))
        ax = seaborn.heatmap([probs], 
        linewidths=.1, square=True, cmap='Greens', vmax=1., cbar_kws={"orientation": "horizontal", "shrink":0.3, "location": "top"})
        ax.set_xticks(np.arange(len(probs))+0.5)
        ax.set_xticklabels(ans_tokens, rotation=30)
        ax.set_yticklabels(['Probs'])
        fig.tight_layout()
    else:
        wrapped_tokens = [ans_tokens[i:i+32] for i in range(0, len(ans_tokens), 32)]
        wrapped_values = [probs[i:i+32] for i in range(0, len(probs), 32)]
        if len(wrapped_tokens[-1]) < 32:
            wrapped_tokens[-1].extend([''] * (32 - len(wrapped_tokens[-1])))
            wrapped_values[-1] = np.concatenate([wrapped_values[-1], -1*np.zeros(32 - len(wrapped_values[-1]))])

        num_subplots = len(wrapped_values)

        vmin = np.min(probs)
        vmax = np.max(probs)
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap='Greens', norm=norm)

        fig, axes = plt.subplots(num_subplots, figsize=(15, 1+1*num_subplots))

        for i, (tokens, vals) in enumerate(zip(wrapped_tokens, wrapped_values)):
            ax = axes[i] if num_subplots > 1 else axes
            seaborn.heatmap([vals], ax=ax, linewidths=.5, square=True, cmap='Greens', vmin=vmin, vmax=vmax, cbar=False)
            ax.set_xticks(np.arange(len(vals))+0.5)
            ax.set_xticklabels(tokens, rotation=30)#, fontsize = 8
            ax.set_yticklabels(['Probs'])

        plt.subplots_adjust(top=1)
        cbar = plt.colorbar(sm, ax=axes.ravel().tolist(), orientation='horizontal', location='top',shrink=0.3)
        cbar.outline.set_visible(False)
        cbar_position = cbar.ax.get_position()
        new_position = [cbar_position.x0, cbar_position.y0, cbar_position.width, 0.02]
        cbar.ax.set_position(new_position)

        fig.set_constrained_layout(True)
        # fig.tight_layout(rect=[0,0,1,0.85])

    return gr.update(value=fig)    

def token_evo_plot(selected_token, ans_logits):

    if not selected_token:
        print("Selected token is empty, returning empty figure and output.")
        return None
    else:
        index, _ = selected_token.split(':', 1)
        index = int(index)
    
    ans_logits = ans_logits[:, index]   # (1+32, vocab_size)
    per_layer_probs = torch.softmax(ans_logits, dim=-1).float().detach().cpu().numpy()   # (1+32, vocab_size)
    per_layer_confs = np.max(per_layer_probs, axis=-1)  # (1+32,)

    per_layer_ids = np.argmax(per_layer_probs, axis=-1)  # (1+32,)
    per_layer_words = [sys.tokenizer.decode(int(id)) for id in per_layer_ids]

    fig = plt.figure(figsize=(15, 2))
    ax = seaborn.heatmap([per_layer_confs], 
       linewidths=.5, square=True, cmap='Blues', vmax=1., cbar_kws={"orientation": "horizontal", "shrink": 0.3, "location": "top"})# 
    ax.set_xticks(np.arange(len(per_layer_confs))+0.5)
    ax.set_xticklabels(per_layer_words, rotation=60)
    ax.set_yticklabels(['Probs'])
    fig.tight_layout()

    return gr.update(value=fig)


def returnfig(qus_tokens, prev_ans_tokens, gr_img, index, relevancy_scores, layer_select, low_th=0.):
        
    gr_img = gr_img.convert('RGB')
    image_tensor = sys.preprocess_image_for_visualize(gr_img)

    fig, ax = plt.subplots()
    ax.imshow(image_tensor.permute(1, 2, 0))
    ax.axis('off')
    qus_tokens = [token.lstrip('▁') for token in qus_tokens]

    R = relevancy_scores[index]
    last_token_scores = relevancy_scores[-1]###########
    first_token_scores = relevancy_scores[0] 

    if layer_select == 0:
        qus_pd = pd.DataFrame({'Tokens': [f'{i}:{token}' for i, token in enumerate(qus_tokens)], 'R': [0] * len(qus_tokens)})
        return fig, qus_pd, None, 'Relevancy not available before layer 0.', 0.5
    
    else:
        r_raw_img = R['raw_img'][layer_select - 1].clone()
        # _, top_indices_first = torch.topk(first_token_scores['raw_img'][layer_select - 1], k=6, largest=True)
        # _, top_indices_last = torch.topk(last_token_scores['raw_img'][layer_select - 1], k=1, largest=True)
        # # top_indices_last = torch.tensor([])
        # combined_indices = torch.unique(torch.cat((top_indices_first, top_indices_last)))

        # raw_scores = last_token_scores['raw_img'][layer_select - 1]  # 获取对应层的一维原始分数
        # normalized_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
        # combined_indices = (normalized_scores > 0.2).nonzero(as_tuple=True)[0]  # 获取满足条件的索引

        # mask = torch.ones_like(r_raw_img, dtype=bool)
        # mask[combined_indices] = False
        # temp_value = r_raw_img[mask].median() 
        # r_raw_img[combined_indices] = temp_value ###########

        r_img = R['img'][layer_select - 1]
        r_qus = R['qus'][layer_select - 1]
        # top_img, _ = r_img.topk(k=topk)
        sum_img = r_img.sum()
        if R['ans'] is not None:
            r_ans = R['ans'][layer_select - 1]
            r_text = torch.cat([r_qus, r_ans])
        else:
            r_text = r_qus
        sum_text = r_text.sum()

        outstr = f'Sum R_img / R_text: {sum_img / sum_text :.2f}'
        max_img = r_raw_img.max().item()
        max_text = r_text.max().item()
        # max_value = max(max_img, max_text)

        h = w = int(sys.model.num_img_patches**0.5)
        reshaped_tensor = r_raw_img.reshape(h, w).unsqueeze(0).unsqueeze(0).float()
        reshaped_tensor[reshaped_tensor < low_th] = 0.
        
        """插值前滤波"""
        reshaped_np = reshaped_tensor.squeeze(0).squeeze(0).detach().cpu().numpy()  # (h, w) 格式
        smoothed_np = cv2.GaussianBlur(reshaped_np, (3, 3), 0)  # 高斯核大小为 (5, 5)
        smoothed_tensor = torch.from_numpy(smoothed_np).unsqueeze(0).unsqueeze(0).to(reshaped_tensor.device)

        interpolated_tensor = F.interpolate(
            smoothed_tensor, size=image_tensor.shape[-2:], mode='bicubic', align_corners=False
        )
        interpolated_tensor = interpolated_tensor.squeeze(0).squeeze(0)

        tensor_np = np.float32(interpolated_tensor.detach().cpu())
        max_tensor = tensor_np.max()
        vis = ax.imshow(tensor_np, cmap='coolwarm', alpha=0.75, vmax=max_tensor)

        # """插值后滤波"""
        # interpolated_tensor = F.interpolate(
        #     reshaped_tensor, size=image_tensor.shape[-2:], mode='bicubic', align_corners=False
        # )
        # interpolated_tensor = interpolated_tensor.squeeze(0).squeeze(0)

        # tensor_np = np.float32(interpolated_tensor.detach().cpu())
        # smoothed_np = cv2.GaussianBlur(tensor_np, (7, 7), 0)
        # max_tensor = smoothed_np.max()
        # vis = ax.imshow(smoothed_np, cmap='coolwarm', alpha=0.75, vmax=max_tensor)

        fig.colorbar(vis, ax=ax)
        
        qus_pd = pd.DataFrame({'Tokens': [f'{i}:{token}' for i, token in enumerate(qus_tokens)], 'R': r_qus.tolist()})
        ans_pd = pd.DataFrame({'Tokens': [f'{i}:{token}' for i, token in enumerate(prev_ans_tokens)], 'R': r_ans.tolist()}) if R['ans'] is not None else None

        return fig, qus_pd, ans_pd, outstr, max_img, max_text

def token_plot(selected_token, relevancy_scores, qus_tokens, ans_tokens, gr_img, layer_select):

    print(f'draw {selected_token} at layer {layer_select}')
    
    if not selected_token:
        print("Selected token is empty, returning empty figure and output.")
        return None, None, None, None, None, None
    else:
        index, _ = selected_token.split(':', 1)
        index = int(index)
    
    prev_ans_tokens = ans_tokens[:index] if index > 0 else []
    min_img = relevancy_scores[index]['raw_img'][layer_select - 1].min().item()

    fig, qus_pd, ans_pd, outstr, max_img, max_text = returnfig(qus_tokens, prev_ans_tokens, gr_img, index, relevancy_scores, layer_select)
    _step = max_img/20 #0.005

    img_state = gr.update(value=fig)
    qus_state = gr.update(value=qus_pd, y_lim=[0,max_text])
    ans_state = gr.update(value=ans_pd, y_lim=[0,max_text]) if ans_pd is not None else None
    str_state = gr.update(value=outstr)

    return img_state, qus_state, ans_state, str_state, gr.update(maximum=max_img,value=max_img,step=_step), gr.update(maximum=max_img,minimum=min_img,value=min_img,step=_step)


def _normalize_tensor(tensor):
    """归一化到 [0, 1]"""
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())

def _mask_and_replace(r_raw_img, mask_indices):
    """根据索引掩蔽并替换为中位数"""
    r_raw_img_copy = r_raw_img.clone()
    mask = torch.ones_like(r_raw_img_copy, dtype=bool)
    mask[mask_indices] = False
    replacement_value = r_raw_img_copy[mask].median()
    r_raw_img_copy[mask_indices] = replacement_value
    return r_raw_img_copy


def _preprocess_image(gr_img):
    """对输入图片进行预处理"""
    gr_img = gr_img.convert('RGB')
    return sys.preprocess_image_for_visualize(gr_img)

def _apply_gaussian_filter(tensor, kernel_size=(3, 3), sigma=0):

    tensor_np = tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
    smoothed_np = cv2.GaussianBlur(tensor_np, kernel_size, sigma)
    smoothed_tensor = torch.from_numpy(smoothed_np).unsqueeze(0).unsqueeze(0).to(tensor.device)
    return smoothed_tensor
    
def _visualize_tensor(tensor, image_tensor, cmap, alpha=0.75, vmax=None):
    """插值并可视化张量"""

    interpolated_tensor = F.interpolate(
        tensor, size=image_tensor.shape[-2:], mode='bicubic', align_corners=False
    ).squeeze(0).squeeze(0)

    tensor_np = np.float32(interpolated_tensor.detach().cpu())
    fig, ax = plt.subplots()
    ax.imshow(image_tensor.permute(1, 2, 0))
    vis = ax.imshow(tensor_np, cmap=cmap, alpha=alpha, vmax=vmax)
    fig.colorbar(vis, ax=ax)
    ax.axis('off')
    return fig


def deartifact(selected_token, relevancy_scores, gr_img, layer_select, top_k_select, threshold_select):
    index, _ = selected_token.split(':', 1)
    index = int(index)
    
    R = relevancy_scores[index]
    last_token_scores = relevancy_scores[-1]
    first_token_scores = relevancy_scores[0]
    
    image_tensor = _preprocess_image(gr_img)
    r_raw_img = R['raw_img'][layer_select - 1].clone()
    h, w = int(sys.model.num_img_patches**0.5), int(sys.model.num_img_patches**0.5)

    # 方法一：Top-K 筛选
    top_indices_first = torch.topk(first_token_scores['raw_img'][layer_select - 1], k=1, largest=True)[1]
    top_indices_last = torch.topk(last_token_scores['raw_img'][layer_select - 1], k=top_k_select, largest=True)[1]
    combined_indices = torch.unique(torch.cat((top_indices_first, top_indices_last)))


    r_raw_img_top = _mask_and_replace(r_raw_img, combined_indices)
    r_raw_img_top = r_raw_img_top.reshape(h, w).unsqueeze(0).unsqueeze(0).float()

    r_raw_img_top = _apply_gaussian_filter(r_raw_img_top)
    fig_top_k = _visualize_tensor(r_raw_img_top, image_tensor, cmap='coolwarm', alpha=0.75, vmax=r_raw_img_top.max())

    # 方法二：高斯滤波 + 阈值筛选
    raw_scores = last_token_scores['raw_img'][layer_select - 1]
    """归一化"""
    # normalized_scores = _normalize_tensor(raw_scores.clone())
    # threshold_indices = (normalized_scores > threshold_select).nonzero(as_tuple=True)[0]
    """累和占比"""
    sorted_scores, sorted_indices = torch.sort(raw_scores.clone().view(-1), descending=True)
    total_sum = sorted_scores.sum()
    threshold_sum = total_sum * (threshold_select / 100)
    # 通过累加累和，找到超过阈值的索引
    cumulative_sum = torch.cumsum(sorted_scores, dim=0)
    threshold_indices_sorted = (cumulative_sum <= threshold_sum).nonzero(as_tuple=True)[0]
    if threshold_indices_sorted.numel() > 0:
        # 根据排序的索引映射回原始位置的索引
        threshold_indices = sorted_indices[:threshold_indices_sorted[-1] + 1]
    else:
        # 如果没有满足条件的索引，threshold_indices 为空
        threshold_indices = torch.tensor([], dtype=torch.long)

    r_raw_img_ratio = _mask_and_replace(r_raw_img, threshold_indices)
    r_raw_img_ratio = r_raw_img_ratio.reshape(h, w).unsqueeze(0).unsqueeze(0).float()

    r_raw_img_ratio = _apply_gaussian_filter(r_raw_img_ratio)
    fig_gauss = _visualize_tensor(r_raw_img_ratio, image_tensor, cmap='coolwarm', alpha=0.75, vmax=r_raw_img_ratio.max())

    return fig_top_k, fig_gauss


def max_image_relevency_plot(selected_token, relevancy_scores, gr_img, layer_select, max_image_relevency, image_low_th):
    
    index, _ = selected_token.split(':', 1)
    index = int(index)
    low_th = image_low_th
    R = relevancy_scores[index]
    last_token_scores = relevancy_scores[-1]
    first_token_scores = relevancy_scores[0] ###########

    gr_img = gr_img.convert('RGB')
    image_tensor = sys.preprocess_image_for_visualize(gr_img)

    fig, ax = plt.subplots()
    ax.imshow(image_tensor.permute(1, 2, 0))
    ax.axis('off')

    r_raw_img = R['raw_img'][layer_select - 1].clone()
    _, top_indices_first = torch.topk(first_token_scores['raw_img'][layer_select - 1], k=1, largest=True)
    _, top_indices_last = torch.topk(last_token_scores['raw_img'][layer_select - 1], k=6, largest=True)

    combined_indices = torch.unique(torch.cat((top_indices_first, top_indices_last)))

    mask = torch.ones_like(r_raw_img, dtype=bool)
    mask[combined_indices] = False
    temp_value = r_raw_img[mask].median() 
    r_raw_img[combined_indices] = temp_value ###########

    r_img = R['img'][layer_select - 1]
    r_qus = R['qus'][layer_select - 1]
    sum_img = r_img.sum()
    if R['ans'] is not None:
        r_ans = R['ans'][layer_select - 1]
        r_text = torch.cat([r_qus, r_ans])
    else:
        r_text = r_qus
    sum_text = r_text.sum()

    # outstr = f'Sum R_img / R_text: {sum_img / sum_text :.2f}'
    # max_value = max(r_img.max().item(), r_text.max().item())

    h = w = int(sys.model.num_img_patches**0.5)
    reshaped_tensor = r_raw_img.reshape(h, w).unsqueeze(0).unsqueeze(0).float()
    reshaped_tensor[reshaped_tensor < low_th] = low_th

    interpolated_tensor = F.interpolate(reshaped_tensor, size=image_tensor.shape[-2:], mode='bicubic', align_corners=False)
    interpolated_tensor = interpolated_tensor.squeeze(0).squeeze(0)
    tensor_np = np.float32(interpolated_tensor.detach().cpu())

    vis = ax.imshow(tensor_np, cmap='coolwarm', alpha=0.75, vmax=max_image_relevency)
    fig.colorbar(vis, ax=ax)
    img_state = gr.update(value=fig)
    return img_state


def cal_vc(relevancy_scores, vcmode):

    visual_confs = []
    for R in relevancy_scores:
        # (layer_idx, content)
        r_img = R['img']
        r_qus = R['qus']
        # top_img, _ = r_img.topk(k=int(topk), dim=1)
        if vcmode == "sum":
            sum_img = r_img.sum(dim=1)
            if R['ans'] is not None:
                r_ans = R['ans']
                r_text = torch.cat([r_qus, r_ans], dim=1)
                # top_text, _ = r_text.topk(k=int(topk), dim=1)
                sum_text = r_text.sum(dim=1)
            else:
                # top_text, _ = r_qus.topk(k=int(topk), dim=1)
                sum_text = r_qus.sum(dim=1)
            vc = sum_img / sum_text   # (layer_idx,)
            _step = 0.1
            # _value = 3
        elif vcmode == "max":
            max_img = r_img.max(dim=1)[0]
            if R['ans'] is not None:
                r_ans = R['ans']
                r_text = torch.cat([r_qus, r_ans], dim=1)
                max_text = r_text.max(dim=1)[0]
            else:
                max_text = r_qus.max(dim=1)[0]
            vc = max_img / max_text   # (layer_idx,)
            _step = 0.05
            # _value = 0.3
        elif vcmode == "mean":
            mean_img = r_img.mean(dim=1)
            if R['ans'] is not None:
                r_ans = R['ans']
                r_text = torch.cat([r_qus, r_ans], dim=1)
                # top_text, _ = r_text.topk(k=int(topk), dim=1)
                mean_text = r_text.mean(dim=1)
            else:
                # top_text, _ = r_qus.topk(k=int(topk), dim=1)
                mean_text = r_qus.mean(dim=1)
            vc = mean_img / mean_text   # (layer_idx,)
            _step = 0.05
        visual_confs.append(vc)
    
    # ->(num_layers, num_tokens)
    visual_confs = torch.stack(visual_confs, dim=0).detach().cpu().numpy().T
    visual_confs = visual_confs[::-1]
    max_last_layer = visual_confs[0].max()
    
    return visual_confs, gr.update(maximum=max_last_layer, step=_step, value=max_last_layer)

def vc_plot(visual_confs, ans_tokens, layer_select):
    
    all_layers, num_tokens = visual_confs.shape
    value = visual_confs[all_layers - layer_select,:]
    max_current_layer = value.max()
    min_current_layer = value.min()

    if len(ans_tokens)<=32:
        fig = plt.figure(figsize=(15, 2))
        ax = seaborn.heatmap([value], 
            linewidths=.1, square=True, cmap='Reds', vmax=max_current_layer, cbar_kws={"orientation": "horizontal", "shrink":0.3, "location": "top"}
        )
        ax.set_xticks(np.arange(len(value))+0.5)
        ax.set_xticklabels(ans_tokens, rotation=30)
        ax.set_yticklabels([layer_select])
        fig.tight_layout()


    else:
        wrapped_tokens = [ans_tokens[i:i+32] for i in range(0, len(ans_tokens), 32)]
        wrapped_values = [value[i:i+32] for i in range(0, len(value), 32)]
        if len(wrapped_tokens[-1]) < 32: #对齐
            wrapped_tokens[-1].extend([''] * (32 - len(wrapped_tokens[-1])))
            wrapped_values[-1] = np.concatenate( [wrapped_values[-1], -1*np.zeros(32 - len(wrapped_values[-1]))] )  # 使用0进行填充

        num_subplots = len(wrapped_values)

        vmin = np.min(value)
        vmax = np.max(value)
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap='Reds', norm=norm)


        fig, axes = plt.subplots(num_subplots, figsize=(15, 1+1*num_subplots))


        for i, (tokens, vals) in enumerate(zip(wrapped_tokens, wrapped_values)):
            ax = axes[i]
            vals = np.clip(vals, None, vmax)#-1e-9
            seaborn.heatmap([vals], ax=ax, linewidths=.5, square=True, cmap='Reds', vmin=vmin, vmax=vmax, cbar=False)#-1e-8

            ax.set_xticks(np.arange(len(vals))+0.5)
            ax.set_xticklabels(tokens, rotation=30)
            ax.set_yticklabels([layer_select])

        plt.subplots_adjust(top=1)
        cbar = plt.colorbar(sm, ax=axes.ravel().tolist(), orientation='horizontal', location='top',shrink=0.3)
        cbar.outline.set_visible(False)
        cbar_position = cbar.ax.get_position()
        new_position = [cbar_position.x0, cbar_position.y0, cbar_position.width, 0.02]
        cbar.ax.set_position(new_position)

        fig.set_constrained_layout(True)
        # fig.tight_layout(rect=[0,0,1,0.85])

    return gr.update(value=fig), gr.update(maximum=max_current_layer, minimum=min_current_layer, value=max_current_layer)


def max_vc_plot(visual_confs, ans_tokens, max_vc, layer_select):

    all_layers, num_tokens = visual_confs.shape
    value = visual_confs[all_layers - layer_select,:]

    if len(ans_tokens) <= 32:
        fig = plt.figure(figsize=(15, 2))
        ax = seaborn.heatmap([value], 
            linewidths=.1, square=True, cmap='Reds', vmax=max_vc, cbar_kws={"orientation": "horizontal", "shrink":0.3, "location": "top"}
        )
        ax.set_xticks(np.arange(len(value))+0.5)
        ax.set_xticklabels(ans_tokens, rotation=30)
        ax.set_yticklabels([layer_select])
        fig.tight_layout()

    else:
        wrapped_tokens = [ans_tokens[i:i+32] for i in range(0, len(ans_tokens), 32)]
        wrapped_values = [value[i:i+32] for i in range(0, len(value), 32)]
        if len(wrapped_tokens[-1]) < 32: #对齐
            wrapped_tokens[-1].extend([''] * (32 - len(wrapped_tokens[-1])))
            wrapped_values[-1] = np.concatenate([wrapped_values[-1], -1*np.zeros(32 - len(wrapped_values[-1]))])

        num_subplots = len(wrapped_values)

        vmin = np.min(value)
        vmax = max_vc
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap='Reds', norm=norm)

        fig, axes = plt.subplots(num_subplots, figsize=(15, 1+1*num_subplots))

        for i, (tokens, vals) in enumerate(zip(wrapped_tokens, wrapped_values)):
            ax = axes[i]
            vals = np.clip(vals, None, vmax-1e-9)
            seaborn.heatmap([vals], ax=ax, linewidths=.5, square=True, cmap='Reds', vmin=vmin-1e-8, vmax=vmax, cbar=False)
            ax.set_xticks(np.arange(len(vals))+0.5)
            ax.set_xticklabels(tokens, rotation=30)
            ax.set_yticklabels([layer_select])

        plt.subplots_adjust(top=1)
        cbar = plt.colorbar(sm, ax=axes.ravel().tolist(), orientation='horizontal', location='top',shrink=0.3)
        cbar.outline.set_visible(False)
        cbar_position = cbar.ax.get_position()
        new_position = [cbar_position.x0, cbar_position.y0, cbar_position.width, 0.02]
        cbar.ax.set_position(new_position)

        fig.set_constrained_layout(True)
        # fig.tight_layout(rect=[0,0,1,0.85])

    return gr.update(value=fig)

def pca_plot(pca_tokens, pca_layer_select):
    t = pca_tokens[pca_layer_select]   # [n_img, 3]

    t_min = t.min(dim=0, keepdim=True).values
    t_max = t.max(dim=0, keepdim=True).values
    normalized_t = (t - t_min) / (t_max - t_min)

    array = (normalized_t * 255).byte().numpy()
    h = w = int(sys.model.num_img_patches**0.5)
    array = array.reshape(h, w, 3)

    fig, ax = plt.subplots()
    ax.imshow(array)
    ax.axis('off')
    return gr.update(value=fig)


title = """<h1 align="center">Demo of LVLM Interpretability</h1>"""

#TODO show examples below

with gr.Blocks(
    # theme=gr.themes.Default(primary_hue=gr.themes.colors.emerald, secondary_hue=gr.themes.colors.green)
) as demo:
    
    gr.Markdown(title)

    inputs = gr.State()
    relevancy_scores = gr.State()
    probs = gr.State()
    qus_tokens = gr.State()
    ans_tokens = gr.State()
    ans_logits = gr.State()
    visual_confs = gr.State()
    pca_tokens = gr.State()
    # gr_img = gr.State()


    gr.Markdown("""<h3>Chat Box</h3>""")
    with gr.Row():
        with gr.Column(scale=2):
            upload_button = gr.Button(scale=1,value="Upload & Start Chat", interactive=False, variant="primary")
            gr_img = gr.Image(scale=3,type="pil")
        with gr.Column(scale=4):
            model_name = args.model_name.split('/')[-1]
            chatbot = gr.Chatbot(label=f'{model_name} (single-round conversion)')
            with gr.Row():
                with gr.Column(scale=1):
                    temperature = gr.Slider(label='Temperature', minimum=0, maximum=1, step=0.1, value=0, interactive=True)
                with gr.Column(scale=1):
                    num_beams = gr.Slider(label='Num Beams', minimum=1, maximum=5, step=1, value=1, interactive=True)
            text_input = gr.Textbox(label='User', placeholder='Please upload your image first.', interactive=False)

    gr.Markdown("""<h3>Visual Contribution</h3>""")
    with gr.Row():
        with gr.Column(scale=1):
            vc_layer_select = gr.Slider(label='LLM Layer', minimum=1, maximum=32, step=1, value=32, interactive=True)
        with gr.Column(scale=1):
            vcmode = gr.Dropdown(label='Calculation mode', choices=["mean", "sum", "max"], value="mean")
        with gr.Column(scale=1):
            max_vc = gr.Slider(label='Max vc', minimum=0, maximum=3, step=0.1, value=3, interactive=True)
    with gr.Row():
        vc_img = gr.Plot(min_width=80, scale=1, label='Per token visual contribution')
    
    gr.Markdown("""<h3>Answer Probabilities</h3>""")
    with gr.Row():
        token_prob = gr.Plot(min_width=80, scale=1, label='')

    gr.Markdown("""<h3>Answer Tokens for Selection</h3>""")
    with gr.Row():
        ans_tokens_to_select = gr.Radio(choices=[], label="Tokens", info="Select one token from the answer.")
    
    gr.Markdown("""<h3>Prediction Evolution in LLM</h3>""")
    with gr.Row():
        token_evo = gr.Plot(min_width=80, scale=1, label='')
    
    gr.Markdown("""<h3>Relevancy Analysis</h3>""")
    with gr.Row():
        with gr.Column(scale=2):
            layer_select = gr.Slider(label='LLM Layer', minimum=1, maximum=40, step=1, value=32, interactive=True)
            result_img = gr.Plot(label='Relevancy to Image')
            max_image_relevency = gr.Slider(label='max image relevency', minimum=0, step=0.001, interactive=True)
            image_low_th = gr.Slider(label='image_low_th', minimum=0, maximum=0.1, step=0.001, interactive=True)
            printstr = gr.Textbox(label='Visual Contribution')
        with gr.Column(scale=4):
            result_qus = gr.BarPlot(scale=1, x='Tokens', y='R', label='Relevancy to User Input', tooltip=['Tokens','R'], interactive=True) # height=145,
            result_ans = gr.BarPlot(scale=1, x='Tokens', y='R', label='Relevancy to Previous Answer', tooltip=['Tokens','R'], interactive=True)# height=145,
    
    gr.Markdown("""<h3>PCA Analysis</h3>""")
    with gr.Row():
        with gr.Column(scale=1):
            pca_layer_select = gr.Slider(label='LLM Layer', minimum=0, maximum=40, step=1, value=32, interactive=True)
            pca_img = gr.Plot(label='PCA after ViT')
        with gr.Column(scale=1):
            # pca_vit = gr.Plot(label='xxxx')
            top_k_select = gr.Slider(label='top_k', minimum=0, maximum=20, step=1, value=5, interactive=True)
            result_img_top_k = gr.Plot(label='w/o fisrt token top1 + last token top_k')
        with gr.Column(scale=1):
            # pca_llm = gr.Plot(label='xxxx')
            threshold_select = gr.Slider(label='percent', minimum=0, maximum=100, value=16, step=2, interactive=True)
            # threshold_select = gr.Slider(label='threshold', minimum=0.05, maximum=1, step=0.05, value=0.2, interactive=True)
            result_img_ratio = gr.Plot(label='normalization + exceed threshold')



    """image and text_input"""

    gr_img.upload(
        upload_img,
        [gr_img],
        [gr_img, text_input, upload_button]
    )
    
    text_input.submit(
        gradio_ask, 
        [chatbot, text_input],
        [chatbot, result_img, result_qus, result_ans, vc_img, token_prob, token_evo]
    ).then(
        gradio_answer,
        [chatbot, gr_img, text_input, temperature, num_beams],
        [chatbot, ans_tokens_to_select, relevancy_scores, probs, qus_tokens, ans_tokens, ans_logits, pca_tokens]
    ).then(
        cal_vc,
        [relevancy_scores, vcmode],
        [visual_confs, max_vc]
    ).then(
        vc_plot,
        [visual_confs, ans_tokens, vc_layer_select],
        [vc_img, max_vc]
    ).then(
        prob_plot,
        [ans_tokens, probs],
        [token_prob]
    )

    """visual confidence"""
    vc_layer_select.release(
        vc_plot,
        [visual_confs, ans_tokens, vc_layer_select],
        [vc_img, max_vc]
    )
    vcmode.change(
        cal_vc,
        [relevancy_scores, vcmode],
        [visual_confs, max_vc]
    ).then(
        vc_plot,
        [visual_confs, ans_tokens, vc_layer_select],
        [vc_img, max_vc]
    )
    max_vc.release(
        max_vc_plot,
        [visual_confs, ans_tokens, max_vc, vc_layer_select],
        [vc_img]
    )
    
    """token and layer"""
    ans_tokens_to_select.select(
        token_plot,
        [ans_tokens_to_select, relevancy_scores, qus_tokens, ans_tokens, gr_img, layer_select], 
        [result_img, result_qus, result_ans, printstr, max_image_relevency, image_low_th]
    ).then(
        token_evo_plot,
        [ans_tokens_to_select, ans_logits],
        [token_evo]
    ).then(
        deartifact,
        [ans_tokens_to_select, relevancy_scores, gr_img, layer_select, top_k_select, threshold_select],
        [result_img_top_k, result_img_ratio]
    )

    top_k_select.release(
        deartifact,
        [ans_tokens_to_select, relevancy_scores, gr_img, layer_select, top_k_select, threshold_select],
        [result_img_top_k, result_img_ratio]
    )

    threshold_select.release(
        deartifact,
        [ans_tokens_to_select, relevancy_scores, gr_img, layer_select, top_k_select, threshold_select],
        [result_img_top_k, result_img_ratio]
    )

    layer_select.release(
        token_plot,
        [ans_tokens_to_select, relevancy_scores, qus_tokens, ans_tokens, gr_img, layer_select], 
        [result_img, result_qus, result_ans, printstr, max_image_relevency, image_low_th]
    )
    max_image_relevency.release(
        max_image_relevency_plot,
        [ans_tokens_to_select, relevancy_scores, gr_img, layer_select, max_image_relevency, image_low_th],
        [result_img]
    )
    image_low_th.release(
        max_image_relevency_plot,
        [ans_tokens_to_select, relevancy_scores, gr_img, layer_select, max_image_relevency, image_low_th],
        [result_img]
    )


    pca_layer_select.release(
        pca_plot,
        [pca_tokens, pca_layer_select],
        [pca_img]
    )


    gr_img.clear(
        gradio_reset, 
        [], 
        [chatbot, text_input, 
         upload_button, ans_tokens_to_select, layer_select, 
         token_prob, result_img, result_qus, result_ans, printstr, vc_img, relevancy_scores, probs, token_evo,
         max_vc],
        queue=False
    )

    gr_img.clear(
        gradio_reset, 
        [], 
        [chatbot, text_input],
        queue=False
    )


demo.queue(max_size=10)
demo.launch(inbrowser=True, share=SHARE)
