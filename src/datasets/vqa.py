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
from . import DataArguments
from ..util.pipeline import ItrDataPipeline, LstDataPipeline
from ..util.pipeline.lst import ItrToLst
from ..util.sym import sym_tbl


class CoCoZhVQAIterDataset(ItrDataPipeline):
    def __init__(
        self,
        text_path: str,
        img_path: str,
        img_slot_size: int,
        img_slot_place: str,
        text_processor: PreTrainedTokenizer,
        img_processor: BlipImageProcessor,
    ) -> None:
        super().__init__(datapipe=[])
        self.text_path = text_path
        self.img_path = img_path
        self.img_slot_size = img_slot_size
        self.img_slot_place = img_slot_place
        self.tokenizer = text_processor
        self.img_processor = img_processor

    def __iter__(self):
        with Path(self.text_path).open() as rf:
            anns = json.load(rf)
        for ann in anns:
            yield self.process_item(ann)

    def process_item(self, ann: Dict[str, Any]):
        image_path = os.path.join(self.img_path, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.img_processor(image, return_tensors="np").pixel_values

        if self.img_slot_place == "prefix":
            # The input format is <img><question><gMask><bos><caption></s>
            question_ids = [
                self.tokenizer.unk_token_id
            ] * self.img_slot_size + self.tokenizer(
                ann["question"], add_special_tokens=False
            )[
                "input_ids"
            ]
        elif self.img_slot_place == "suffix":
            question_ids = (
                self.tokenizer(ann["question"], add_special_tokens=False)["input_ids"]
                + [self.tokenizer.unk_token_id] * self.img_slot_size
            )
        else:
            raise ValueError(f"Invalid img_slot_place: {self.img_slot_place}")
        caption_ids = self.tokenizer(ann["answer"], add_special_tokens=False)[
            "input_ids"
        ]

        # make a copy of question_ids. build_inputs_with_special_tokens will modify it
        input_ids = self.tokenizer.build_inputs_with_special_tokens(
            question_ids, caption_ids
        )
        context_length = input_ids.index(self.tokenizer.bos_token_id)
        image_slot_offset = input_ids.index(self.tokenizer.unk_token_id)

        input_ids = np.asarray(input_ids)
        labels = np.copy(input_ids)
        labels[:context_length] = -100

        return {
            "image": image,  # (3, 224, 224) maybe
            "input_ids": input_ids,  # (seq_len), image included
            "image_slot_offset": image_slot_offset,
            "labels": labels,  # (seq_len), image included
        }


class CoCoZhVQADataset(LstDataPipeline):
    @classmethod
    def load_dataset(
        cls, split: str, image_size: int, num_query_tokens: int
    ) -> Optional[Dataset]:
        data_args: DataArguments = sym_tbl().cfg["data"]

        if split == "train":
            text_path = data_args.train_data_path
        elif split == "dev":
            text_path = data_args.dev_data_path
            if len(text_path) == 0:
                return None
        else:
            raise ValueError(f"Invalid split: {split}")

        tokenizer = AutoTokenizer.from_pretrained(
            sym_tbl().cfg["model"].blip2_path, trust_remote_code=True
        )
        image_processor = BlipImageProcessor(
            size={"height": image_size, "width": image_size},
            image_mean=OPENAI_CLIP_MEAN,
            image_std=OPENAI_CLIP_STD,
        )

        return cls(
            text_path=text_path,
            img_path=data_args.img_data_path,
            img_slot_size=num_query_tokens,
            img_slot_place=data_args.img_slot_place,
            text_processor=tokenizer,
            img_processor=image_processor,
        )

    def __init__(
        self,
        text_path: str,
        img_path: str,
        img_slot_size: int,
        img_slot_place: str,
        text_processor: PreTrainedTokenizer,
        img_processor: BlipImageProcessor,
    ) -> None:
        super().__init__(
            datapipe=ItrToLst(
                CoCoZhVQAIterDataset(
                    text_path=text_path,
                    img_path=img_path,
                    img_slot_size=img_slot_size,
                    img_slot_place=img_slot_place,
                    text_processor=text_processor,
                    img_processor=img_processor,
                )
            )
        )

    def __getitem__(self, index):
        return self.datapipe[index]

    def __len__(self) -> int:
        return len(self.datapipe)
