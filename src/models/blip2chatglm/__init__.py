from transformers import Trainer, PreTrainedModel, AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
import os
from .modeling_blip2chatglm import Blip2ChatGLMConfig, Blip2ChatGLM, Blip2ForChatGLM
from .modeling_chatglm import ChatGLMForConditionalGeneration
from .dataset import (
    Blip2ChatGLMDataArguments,
    MEPAVEDataset,
    mepave_collator,
)
from ...util.sym import sym_tbl


@dataclass
class Blip2ChatGLMModelArguments:
    blip2_path: str = field(default="models/blip2zh-chatglm-6b")
    lm_path: str = field(default="models/chatglm-6b")
    lora_rank: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


class Blip2ChatGLMTrainer(Trainer):
    @classmethod
    def load_model(cls) -> PreTrainedModel:
        model_args: Blip2ChatGLMModelArguments = sym_tbl().cfg["model"]

        lm = ChatGLMForConditionalGeneration.from_pretrained(
            model_args.lm_path,
            # load_in_8bit=True,
            # device_map="auto",
        )
        # lm.gradient_checkpointing_enable()
        lm.enable_input_require_grads()
        # lm.is_parallelizable = True
        # lm.model_parallel = True
        lm.lm_head = CastOutputToFloat(lm.lm_head)
        lm.config.use_cache = (
            False  # silence the warnings. Please re-enable for inference!
        )

        # setup peft
        # peft_config = LoraConfig(
        #     task_type=TaskType.CAUSAL_LM,
        #     inference_mode=False,
        #     r=model_args.lora_rank,
        #     lora_alpha=model_args.lora_alpha,
        #     lora_dropout=model_args.lora_dropout,
        # )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=[
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h",
            ],
        )
        lm = get_peft_model(lm, peft_config)
        lm.print_trainable_parameters()

        blip2 = Blip2ForChatGLM.from_pretrained(
            model_args.blip2_path,
        )
        for param in blip2.parameters():
            # Freeze blip2
            param.requires_grad = False

        blip2_config = Blip2ChatGLMConfig.from_pretrained(model_args.blip2_path)

        return Blip2ChatGLM(blip2_config, blip2, lm)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs["input_ids"],
            context_masks=inputs["context_masks"],
            input_ids_masks=inputs["input_ids_masks"],
            images=inputs["images"],
            labels=inputs["labels"],
            return_dict=True,
        )
        if return_outputs:
            return outputs.loss, outputs
        else:
            return outputs.loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        # only save lora (requires_grad)
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))


__all__ = [
    "Blip2ChatGLMTrainer",
    "Blip2ChatGLMModelArguments",
    "Blip2ChatGLMDataArguments",
    "MEPAVEDataset",
    "mepave_collator",
]
