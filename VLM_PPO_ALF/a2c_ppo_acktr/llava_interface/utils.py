from dataclasses import dataclass, field
from typing import Optional
import os
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from llava.model import LlavaLlamaForCausalLM


import torch
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_dino_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")

model_args = ModelArguments()
model_args.model_name_or_path = "lmsys/vicuna-7b-v1.5"
model_args.version = "v1"
# model_args.freeze_backbone =
model_args.tune_mm_mlp_adapter = True
model_args.vision_tower = "openai/clip-vit-large-patch14-336"
model_args.mm_vision_select_layer = -2
model_args.pretrain_mm_mlp_adapter = None
model_args.mm_projector_type = "mlp2x_gelu"
model_args.mm_use_im_start_end =  False
model_args.mm_use_im_patch_token = False


def load_lora_model(model_path, model_base = "liuhaotian/llava-v1.5-7b",load_8bit=False, load_4bit=False, cache_dir=None):

    kwargs = {}
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, cache_dir=cache_dir)
    print('Loading LLaVA from base model...')
    model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, cache_dir=cache_dir, **kwargs)
    token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
    if model.lm_head.weight.shape[0] != token_num:
        model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
        model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

    print('Loading additional LLaVA weights...')
    if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
    else:
        # this is probably from HF Hub
        from huggingface_hub import hf_hub_download
        def load_from_hf(repo_id, filename, subfolder=None):
            cache_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder)
            return torch.load(cache_file, map_location='cpu')
        non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
    non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
    if any(k.startswith('model.model.') for k in non_lora_trainables):
        non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
    model.load_state_dict(non_lora_trainables, strict=False)

    from peft import PeftModel
    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, model_path)
    print('Merging LoRA weights...')
    model = model.merge_and_unload()
    print('Model is loaded...')

    return model, tokenizer



def init_pretrained_model(model, tokenizer, pretrain_mm_adapter=None):

    if pretrain_mm_adapter is not None:
        #print("I loaded right projector")
        mm_projector_weights = torch.load(os.path.join(pretrain_mm_adapter, 'mm_projector.bin'), map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        model.load_state_dict(mm_projector_weights, strict=False)
    else:
        pass

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    assert model.config.vocab_size == len(tokenizer), f"vocab size mismatch: {model.config.vocab_size} vs {len(tokenizer)}, need to resize_token_embeddings"

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=model.device, dtype=torch.float16)


    return model, tokenizer

def find_all_linear_names(model,train_vision=None):
    """
        train_vision decides which components from the VLM / MLLM we want to train.
            1. train_vision == 'all' will train vision_tower, mm_projector, and LLM
            2. train_vision == 'projector' will train mm_projector and LLM
            3. otherwise will only train the LLM 
    """
    cls = torch.nn.Linear
    lora_module_names = set()
    if train_vision == 'all':
        multimodal_keywords = ['vision_resampler']
    elif train_vision == 'projector':
        multimodal_keywords = ['vision_resampler', 'vision_tower']
    else:
        multimodal_keywords = ['vision_resampler', 'vision_tower', 'mm_projector']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if "value_head" in name:
            # value head should not be part of the adapter
            continue
        if isinstance(module, cls):
            names = name.split('.')

            if "0"<=names[-1] and names[-1]<="9":
                lora_module_names.add(names[0] if len(names) == 1 else names[-2]+"."+names[-1])
            else:
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    print(list(lora_module_names))
    return list(lora_module_names)
