from dataclasses import dataclass, field
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
from torch.utils.data import IterableDataset, Dataset
from transformers import AutoTokenizer, BlipImageProcessor, PreTrainedTokenizer
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
import torch
from PIL import Image
from .modeling_blip2chatglm import ChatGLMConfig
from ...util.pipeline import ItrDataPipeline, LstDataPipeline
from ...util.pipeline.lst import ItrToLst
from ...util.sym import sym_tbl


@dataclass
class Blip2ChatGLMDataArguments:
    train_data_path: str = field(default="data/alpaca")
    dev_data_path: str = field(default="")
    img_data_path: str = field(default="data/alpaca")
    question: str = field(default="")


class MEPAVEIterDataset(ItrDataPipeline):
    def __init__(
        self,
        text_path: str,
        img_path: str,
        question: str,
        img_slot_size: int,
        text_processor: PreTrainedTokenizer,
        img_processor: BlipImageProcessor,
    ) -> None:
        super().__init__(datapipe=[])
        self.text_path = text_path
        self.img_path = img_path
        self.question = question
        self.img_slot_size = img_slot_size
        self.tokenizer = text_processor
        self.img_processor = img_processor
        self.question_ids = self.tokenizer(question, add_special_tokens=False)[
            "input_ids"
        ]

    def __iter__(self):
        with Path(self.text_path).open() as rf:
            for line in rf:
                ann = json.loads(line)
                yield self.process_item(ann)

    def process_item(self, ann: Dict[str, Any]):
        image_path = os.path.join(self.img_path, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.img_processor(image, return_tensors="np").pixel_values

        # The input format is <img><question><gMask><bos><caption></s>
        caption_ids = self.tokenizer(ann["caption"], add_special_tokens=False)[
            "input_ids"
        ]

        # make a copy of question_ids. build_inputs_with_special_tokens will modify it
        input_ids = self.tokenizer.build_inputs_with_special_tokens(
            list(self.question_ids), caption_ids
        )
        context_length = input_ids.index(self.tokenizer.bos_token_id)

        input_ids = np.asarray(input_ids)
        context_mask = np.zeros(len(input_ids), dtype=bool)
        context_mask[:context_length] = True
        labels = np.copy(input_ids)
        labels[:context_length] = -100

        return {
            "image": image,  # (3, 224, 224) maybe
            "input_ids": input_ids,  # (text_len), image not included
            "context_mask": context_mask,  # (text_len), true for context
            "labels": labels,  # (text_len), image not included
        }


class MEPAVEDataset(LstDataPipeline):
    @classmethod
    def load_dataset(cls, split: str = "train") -> Optional[Dataset]:
        data_args: Blip2ChatGLMDataArguments = sym_tbl().cfg["data"]

        if split == "train":
            text_path = data_args.train_data_path
        elif split == "dev":
            text_path = data_args.dev_data_path
            if len(text_path) == 0:
                return None
        else:
            raise ValueError(f"Invalid split: {split}")

        tokenizer = AutoTokenizer.from_pretrained(
            sym_tbl().cfg["model"].lm_path, trust_remote_code=True
        )
        model_cfg = ChatGLMConfig.from_pretrained(sym_tbl().cfg["model"].blip2_path)
        image_size = model_cfg.vision_config["image_size"]
        image_processor = BlipImageProcessor(
            size={"height": image_size, "width": image_size},
            image_mean=OPENAI_CLIP_MEAN,
            image_std=OPENAI_CLIP_STD,
        )

        return cls(
            text_path,
            data_args.img_data_path,
            data_args.question,
            model_cfg.num_query_tokens,
            tokenizer,
            image_processor,
        )

    def __init__(
        self,
        text_path: str,
        img_path: str,
        question: str,
        img_slot_size: int,
        text_processor: PreTrainedTokenizer,
        img_processor: BlipImageProcessor,
    ) -> None:
        super().__init__(
            datapipe=ItrToLst(
                MEPAVEIterDataset(
                    text_path,
                    img_path,
                    question,
                    img_slot_size,
                    text_processor,
                    img_processor,
                )
            )
        )

    def __getitem__(self, index):
        return self.datapipe[index]

    def __len__(self) -> int:
        return len(self.datapipe)


def mepave_collator(features: list) -> dict:
    longest = max(len(f["input_ids"]) for f in features)
    input_ids = torch.zeros(len(features), longest, dtype=torch.long)
    labels = torch.ones(len(features), longest, dtype=torch.long) * -100
    context_masks = torch.zeros(len(features), longest, dtype=torch.bool)
    images = []
    input_ids_masks = torch.zeros(len(features), longest, dtype=torch.bool)

    for i, feature in enumerate(features):
        input_ids[i, : len(feature["input_ids"])] = torch.as_tensor(
            feature["input_ids"]
        )
        context_masks[i, : len(feature["context_mask"])] = torch.as_tensor(
            feature["context_mask"]
        )
        labels[i, : len(feature["labels"])] = torch.as_tensor(feature["labels"])
        images.append(torch.as_tensor(feature["image"]))
        input_ids_masks[i, : len(feature["input_ids"])] = 1
    return {
        "input_ids": input_ids,
        "context_masks": context_masks,
        "input_ids_masks": input_ids_masks,
        "labels": labels,
        "images": torch.cat(images, dim=0),
    }
