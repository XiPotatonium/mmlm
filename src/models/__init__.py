from dataclasses import dataclass
from typing import Dict, List, Any
from transformers import Trainer
from torch.utils.data import Dataset
from .blip2chatglm import *


@dataclass
class ModelProto:
    trainer: Trainer
    dataset: Dataset
    collate_fn: callable
    args: Dict[str, Any]


MAPPING = {
    "blip2chatglm": ModelProto(
        trainer=Blip2ChatGLMTrainer,
        dataset=MEPAVEDataset,
        collate_fn=mepave_collator,
        args={
            "model": Blip2ChatGLMModelArguments,
            "data": Blip2ChatGLMDataArguments,
        }
    ),
}