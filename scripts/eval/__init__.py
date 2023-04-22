from typing import Optional
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from transformers import CLIPProcessor, AutoTokenizer, BlipImageProcessor, AutoModel
from peft import PeftModel, LoraConfig, get_peft_model
import torch
import os


def load_blip2chatglm(blip2_path: str, lora_path: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(blip2_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        blip2_path, trust_remote_code=True,
    )
    model.setup_dtype(vision_encoder_dtype="fp16", lm_dtype="fp16")
    if lora_path is not None:
        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            lora_path,
            # torch_dtype=torch.float16,
        )
    model.eval()
    image_size = model.config.vision_config.image_size
    image_processor = BlipImageProcessor(
        size={"height": image_size, "width": image_size},
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
    )
    return tokenizer, image_processor, model
