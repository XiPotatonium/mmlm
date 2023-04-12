import json
import re
import sys
from typing import List, Tuple, Union
from loguru import logger
import numpy as np
import torch
from torch import Tensor
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from pathlib import Path
import typer
from torchvision import transforms
from transformers import CLIPProcessor, AutoTokenizer, BlipImageProcessor
from rich.progress import Progress

from src.util.extention.rich import no_total_columns, full_columns
from src.util.device import alloc1
from . import load_blip2chatglm
from ..models.bert_clip import BertCLIPModel


app = typer.Typer()


def get_coco_imgids(folder: Path):
    mapping = {}
    for f in folder.iterdir():
        _, split, imgid = f.stem.split("_")
        mapping[int(imgid)] = f
    return mapping


def pre_caption(caption, max_words: int):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        " ",
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    # truncate caption
    caption_words = caption.split(" ")
    if len(caption_words) > max_words:
        caption = "".join(caption_words[:max_words])

    return caption


def load_laion_dataset(files: str, image_size: int = 224, max_words: int = 64):
    import webdataset as wds

    def to_dict(sample):
        return {
            "image": sample[0],
            "text_input": pre_caption(sample[1]["caption"], max_words),
        }

    # transform_eval = transforms.Compose(
    #     [
    #         transforms.Resize(
    #             (image_size, image_size), interpolation=InterpolationMode.BICUBIC
    #         ),
    #         transforms.ToTensor(),
    #         transforms.Normalize(OPENAI_CLIP_MEAN, OPENAI_CLIP_STD),
    #     ]
    # )

    datapipe = wds.DataPipeline(
        wds.SimpleShardList(files),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        # wds.shuffle(1000, handler=wds.warn_and_continue),
        wds.decode("pilrgb", handler=wds.warn_and_continue),
        wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
        wds.map_tuple(transforms.ToTensor(), handler=wds.warn_and_continue),
        wds.map(to_dict, handler=wds.warn_and_continue),
    )
    return torch.utils.data.DataLoader(datapipe, batch_size=1)


def my_clip_score(
    images: Union[Tensor, List[Tensor]],
    text: Union[str, List[str]],
    model: BertCLIPModel,
    processor: CLIPProcessor,
) -> Tuple[Tensor, int]:
    if not isinstance(images, list):
        if images.ndim == 3:
            images = [images]
    else:  # unwrap into list
        images = [i for i in images]

    if not all(i.ndim == 3 for i in images):
        raise ValueError(
            "Expected all images to be 3d but found image that has either more or less"
        )

    if not isinstance(text, list):
        text = [text]

    if len(text) != len(images):
        raise ValueError(
            f"Expected the number of images and text examples to be the same but got {len(images)} and {len(text)}"
        )
    device = images[0].device
    processed_input = processor(
        text=text, images=[i.cpu() for i in images], return_tensors="pt", padding=True
    )  # type:ignore

    img_features = model.get_image_features(processed_input["pixel_values"].to(device))
    img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

    txt_features = model.get_text_features(
        processed_input["input_ids"].to(device),
        processed_input["attention_mask"].to(device),
    )
    txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)

    # cosine similarity between feature vectors
    score = 100 * (img_features * txt_features).sum(axis=-1)

    score = score.mean(0)
    return torch.max(score, torch.zeros_like(score))


def _score_stat(results: List[dict]):
    scores = [r["score"] for r in results]
    print(f"min: {np.min(scores):.2f}")
    print(f"max: {np.max(scores):.2f}")
    print(f"mean: {np.mean(scores):.2f}")


@app.command()
def testset(
    files: str = typer.Argument(
        "data/laion/laion2b_chinese_release/sim0.34-dev/00000.tar"
    ),
    output: str = typer.Option(...),
    clip_model_path: str = typer.Option("models/clip-vit-bert-chinese-1M"),
):
    clip_model = BertCLIPModel.from_pretrained(clip_model_path)
    CLIPProcessor.tokenizer_class = "BertTokenizerFast"
    clip_proc = CLIPProcessor.from_pretrained(clip_model_path)
    device_info = alloc1([])
    device = torch.device(device_info["device"])
    clip_model.to(device)
    clip_model.eval()

    dataloader = load_laion_dataset(files, image_size=224)
    results = []
    output: Path = Path(output)
    output.parent.mkdir(exist_ok=True, parents=True)
    with torch.no_grad(), Progress(*no_total_columns()) as pbar, output.open(
        "w", encoding="utf8"
    ) as wf:
        tid = pbar.add_task("ClipScore", total=None)
        for sample in dataloader:
            # print(sample["text_input"])
            # print(sample["image"].shape)
            score = my_clip_score(
                sample["image"].to(device), sample["text_input"], clip_model, clip_proc
            )
            result = {
                "score": score.item(),
                "label": sample["text_input"][0],
                "pred": sample["text_input"][0],
            }
            results.append(result)
            wf.write(json.dumps(result, ensure_ascii=False) + "\n")
            pbar.advance(tid)
    _score_stat(results)


@app.command()
def blip2(
    files: str = typer.Argument(
        "data/laion/laion2b_chinese_release/sim0.34-dev/00000.tar"
    ),
    output: str = typer.Option(...),
    clip_model_path: str = typer.Option("models/clip-vit-bert-chinese-1M"),
    blip2_path: str = typer.Option("models/blip2zh-chatglm-6b"),
    lm_path: str = typer.Option("models/chatglm-6b"),
    prompt: str = typer.Option(""),
):
    clip_model = BertCLIPModel.from_pretrained(clip_model_path)
    CLIPProcessor.tokenizer_class = "BertTokenizerFast"
    clip_proc = CLIPProcessor.from_pretrained(clip_model_path)
    tokenizer, blip2_proc, model = load_blip2chatglm(blip2_path, lm_path)
    device_info = alloc1([])
    device = torch.device(device_info["device"])
    clip_model.to(device)
    model.to(device)
    clip_model.eval()
    model.eval()

    # NOTE: Still slow. I don't know why no performance boost.
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     logger.info("Use torch.compile")
    #     model = torch.compile(model)

    dataloader = load_laion_dataset(files, image_size=224)
    results = []
    output: Path = Path(output)
    output.parent.mkdir(exist_ok=True, parents=True)
    with torch.no_grad(), Progress(*no_total_columns()) as pbar, output.open(
        "w", encoding="utf8"
    ) as wf:
        tid = pbar.add_task("ClipScore", total=None)
        for sample in dataloader:
            # print(sample["text_input"])
            # print(sample["image"].shape)
            pixel_values = blip2_proc(
                sample["image"], return_tensors="pt"
            ).pixel_values.to(device)
            MAX_LENGTH = 256
            output, _ = model.chat(
                tokenizer,
                (prompt, pixel_values),
                history=[],
                max_length=MAX_LENGTH,
            )
            if len(output) >= MAX_LENGTH:
                logger.warning(f"output length {len(output)} >= {MAX_LENGTH}: {output}")
                output = output[:MAX_LENGTH]
            score = my_clip_score(
                sample["image"].to(device), output, clip_model, clip_proc
            )
            result = {
                "score": score.item(),
                "label": sample["text_input"][0],
                "pred": output,
            }
            results.append(result)
            wf.write(json.dumps(result, ensure_ascii=False) + "\n")
            pbar.advance(tid)
    _score_stat(results)


@app.command()
def score_stat(path: str):
    results = []
    with open(path, encoding="utf8") as f:
        for line in f:
            results.append(json.loads(line))
    _score_stat(results)


if __name__ == "__main__":
    app()
