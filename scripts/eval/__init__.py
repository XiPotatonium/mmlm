from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from transformers import CLIPProcessor, AutoTokenizer, BlipImageProcessor

from src.models.blip2chatglm.modeling_chatglm import ChatGLMForConditionalGeneration
from src.models.blip2chatglm.modeling_blip2chatglm import Blip2ChatGLM, Blip2ForChatGLM


def load_blip2chatglm(blip2_path: str, lm_path: str):
    tokenizer = AutoTokenizer.from_pretrained(lm_path, trust_remote_code=True)
    lm = ChatGLMForConditionalGeneration.from_pretrained(
        lm_path,  # device_map="auto"
    )
    lm = lm.half()

    blip2 = Blip2ForChatGLM.from_pretrained(
        blip2_path,
    )

    model = Blip2ChatGLM(blip2, lm)
    model.eval()
    image_size = model.blip2.config.vision_config.image_size
    image_processor = BlipImageProcessor(
        size={"height": image_size, "width": image_size},
        image_mean=OPENAI_CLIP_MEAN,
        image_std=OPENAI_CLIP_STD,
    )
    return tokenizer, image_processor, model
