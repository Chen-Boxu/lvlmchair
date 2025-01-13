import sys
import torch
from transformers import AutoProcessor
from transformers.generation.utils import *
from transformers import BitsAndBytesConfig
# imports modules for registration
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from relevancy_utils import *


class LlavaForAnalysis():

    def __init__(self, args):
        args.model_path = args.model_name
        model_name = get_model_name_from_path(args.model_path)
        if "finetune-lora" in args.model_path:
            model_base = "liuhaotian/llava-v1.5-7b"
        elif "lora" in args.model_path:
            model_base = "lmsys/vicuna-7b-v1.5"
        else:
            model_base = None
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(args.model_name, model_base, model_name, load_8bit=args.load_8bit, load_4bit=args.load_4bit)
        self.args = args
        self.model.num_img_patches = self.model.model.vision_tower.num_patches
        self.model.num_img_tokens = self.model.num_img_patches
        self.model.num_llm_layers = self.model.config.num_hidden_layers
        self.model.lm_head = self.model.lm_head
        self.conv_mode = "llava_v1"
    
    def register_hooks(self):
        self.model.vit_satt, self.model.lm_satt = [], []

        # create hooks to capture attentions and their gradients
        vit_forward_hook = create_hook(self.model.vit_satt)
        lm_forward_hook = create_hook(self.model.lm_satt)

        self.hooks = []
        # register hooks with corresponding locations
        for layer in self.model.base_model.vision_tower.vision_tower.vision_model.encoder.layers[:self.model.config.mm_vision_select_layer+1]:
            self.hooks.append(layer.self_attn.register_forward_hook(vit_forward_hook))
        for layer in self.model.base_model.layers:
            self.hooks.append(layer.self_attn.register_forward_hook(lm_forward_hook))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
    
    def refresh_chat(self):
        self.conv = conv_templates[self.conv_mode].copy()
        self.roles = self.conv.roles

    @torch.no_grad()
    def chat(self, image, text):
        self.refresh_chat()

        image = image.convert("RGB")
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.unsqueeze(0).half().to(self.model.device)

        # message
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + text
        # inp = prompt
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], None)

        conv_prompt = self.conv.get_prompt()
        input_ids = tokenizer_image_token(conv_prompt, self.tokenizer, 
                                          IMAGE_TOKEN_INDEX, 
                                          return_tensors='pt').unsqueeze(0).cuda()
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = ["###"]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        # streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        return_dict = True
    
        outputs = self.model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True if self.args.temperature > 0 else False,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            num_beams=self.args.num_beams,
            max_new_tokens=self.args.max_length,
            # streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            return_dict_in_generate=return_dict,
            output_attentions=return_dict,
            output_hidden_states=return_dict,
            output_scores=return_dict,
        )
        return outputs.sequences[0]

    @torch.enable_grad()
    def forward_with_grads(self, image, text, answer):
        self.refresh_chat()

        image = image.convert("RGB")
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.unsqueeze(0).half().to(self.model.device)

        # message
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + text
        # inp = prompt
        self.conv.append_message(self.conv.roles[0], inp)
        self.conv.append_message(self.conv.roles[1], answer)

        conv_prompt = self.conv.get_prompt()
        input_ids = tokenizer_image_token(conv_prompt, self.tokenizer, 
                                          IMAGE_TOKEN_INDEX, 
                                          return_tensors='pt').unsqueeze(0).cuda() #[:,:-1] ### </s>
        self.qus_tokens = self.tokenizer.tokenize(text)
        
        # "<s> A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWhat is it? ASSISTANT: "
        # idx for next token, end_idx is next_start_idx #################
        self.img_start_idx = 35 ##################
        self.img_end_idx = self.img_start_idx + self.model.num_img_tokens
        self.qus_start_idx = self.img_end_idx + 2   # '\n'
        self.qus_end_idx = self.qus_start_idx + len(self.qus_tokens)
        self.ans_start_idx = self.qus_end_idx + 6   # "ASSISTANT: "
        # <s> [the unknown token] Yes, ... </s> ################
        outputs = self.model(
            input_ids,
            images=image_tensor,
            return_dict=True,
            output_attentions=True,
            output_hidden_states=True,
        )
        # outputs.logits = torch.cat((outputs.logits[0, self.ans_start_idx-2:self.ans_start_idx],
        #                     outputs.logits[0, self.ans_start_idx+1:-2])) #################
        outputs.logits = outputs.logits[0, self.ans_start_idx-1:-2]
        outputs.img_hidden_states = [h[0, self.img_start_idx:self.img_end_idx] for h in outputs.hidden_states]
        # outputs.hidden_states = [
        #     torch.cat((h[0, self.ans_start_idx-2:self.ans_start_idx], h[0, self.ans_start_idx+1:-2]))
        #     for h in outputs.hidden_states
        # ] ############
        outputs.hidden_states = [h[0, self.ans_start_idx-1:-2] for h in outputs.hidden_states]
        return outputs
    
    def compute_relevancy(self, word_idx):
        R_i_i = cal_vit_relevancy(self.model)   # (576, 576)
        R_t_t_per_layer = cal_llm_relevancy(self.model, self.ans_start_idx+word_idx)   # (num_layers, curr_len)
        R = {}
        R['img'] = R_t_t_per_layer[:, self.img_start_idx:self.img_end_idx].cpu()
        R['raw_img'] = torch.matmul(R['img'], R_i_i.cpu())   # (num_layers, 576)
        R['qus'] = R_t_t_per_layer[:, self.qus_start_idx:self.qus_end_idx].cpu()
        R['ans'] = R_t_t_per_layer[:, self.ans_start_idx:].cpu()
        R['top_index'] = []
        for layer_idx in range(R['raw_img'].shape[0]):
            _, top_indices = torch.topk(R['raw_img'][layer_idx], k=10, largest=True)
            R['top_index'].append(top_indices.tolist())
        return R

    def compute_relevancy_cached(self, word_idx, gradcam_cache):
        R_i_i = cal_vit_relevancy(self.model)   # (576, 576)
        R_t_t_per_layer, gradcam_cache = cal_llm_relevancy_cached(self.model, word_idx*self.model.num_llm_layers, gradcam_cache)   # (num_layers, curr_len)
        R = {}
        R['img'] = R_t_t_per_layer[:, self.img_start_idx:self.img_end_idx].cpu()
        R['raw_img'] = torch.matmul(R['img'], R_i_i.cpu())   # (num_layers, 576)
        R['qus'] = R_t_t_per_layer[:, self.qus_start_idx:self.qus_end_idx].cpu()
        R['ans'] = R_t_t_per_layer[:, self.ans_start_idx:].cpu()
        return R, gradcam_cache
    
    def preprocess_image_for_visualize(self, image):
        return self.image_processor.preprocess(image, do_normalize=False, return_tensors='pt')['pixel_values'].squeeze(0)
