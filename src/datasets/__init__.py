from dataclasses import dataclass, field

import torch


@dataclass
class DataArguments:
    dataset: str = field(default="mepave")
    train_data_path: str = field(default="data/alpaca")
    dev_data_path: str = field(default="")
    img_data_path: str = field(default="data/alpaca")
    img_slot_place: str = field(default="prefix")
    question: str = field(default="")
    caption_field: str = field(default="caption")


def blip2chatglm_collator(features: list) -> dict:
    longest = max(len(f["input_ids"]) for f in features)
    input_ids = torch.zeros(len(features), longest, dtype=torch.long)
    image_slot_offset = []
    labels = torch.ones(len(features), longest, dtype=torch.long) * -100
    images = []

    for i, feature in enumerate(features):
        input_ids[i, : len(feature["input_ids"])] = torch.as_tensor(
            feature["input_ids"]
        )
        image_slot_offset.append(feature["image_slot_offset"])
        labels[i, : len(feature["labels"])] = torch.as_tensor(feature["labels"])
        images.append(torch.as_tensor(feature["image"]))
    return {
        "input_ids": input_ids,
        "image_slot_offset": torch.as_tensor(image_slot_offset),
        "labels": labels,
        "images": torch.cat(images, dim=0),
    }
