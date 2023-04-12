import json
import re
import sys
from typing import Any, Dict, List, Tuple, Union
from loguru import logger
import numpy as np
import torch
from torch import Tensor
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from pathlib import Path
import typer
import os
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, AutoTokenizer, BlipImageProcessor
from rich.progress import Progress
from sklearn.metrics import precision_recall_fscore_support as prfs

from src.util.extention.rich import no_total_columns, full_columns
from src.util.device import alloc1
from . import load_blip2chatglm
from ..models.bert_clip import BertCLIPModel


app = typer.Typer()


def load_uie_datatset(file: str, img_root: str):
    with Path(file).open('r', encoding="utf8") as rf:
        for line in rf:
            ann = json.loads(line)
            text = ann["text"]
            image_path = os.path.join(img_root, ann["image"])
            image = Image.open(image_path).convert("RGB")

            yield {
                "image_path": ann["image"],
                "image": image,
                "entities": [{"text": text[ent["start"]:ent["end"], "type": ent["type"]]} for ent in ann["entities"]],
            }


def parse_uie(output: str):
    results = []
    while True:
        m = re.match(r"\s*{\s*(\S+)\s*:\s*(\S+)\s*}\s*", output)
        if m is None:
            break
        output = output[m.end():]
        results.append({"text": m.group(2), "type": m.group(1)})
    return results


def _convert(
    results: List[Dict[str, Any]],
    include_entity_types: bool = True,
    pseudo_entity_type: str = "Entity",
):
    converted_gt, converted_pred = [], []

    for sample_results in results:
        preds = []
        gts = []
        for pred in sample_results["pred"]:
            p_tup = [
                pred["text"],
                pred["type"] if include_entity_types else pseudo_entity_type
            ]

            preds.append(tuple(p_tup))
        for gt in sample_results["gts"]:
            gt_tup = [
                gt["text"],
                gt["type"] if include_entity_types else pseudo_entity_type
            ]
            gts.append(tuple(gt_tup))

        converted_gt.append(gts)
        converted_pred.append(preds)

    return converted_gt, converted_pred


def _score(t2id: Dict[str, int], gt: List[List[Tuple]], pred: List[List[Tuple]], cls_metric=False):
    assert len(gt) == len(pred)

    gt_flat = []
    pred_flat = []
    types = set()

    for (sample_gt, sample_pred) in zip(gt, pred):
        union = set()
        if cls_metric:
            union.update(sample_gt)
            text_gt = list(map(lambda x: x[0], sample_gt))
            sample_text_true_pred = list(filter(lambda x: x[0] in text_gt, sample_pred))
            union.update(sample_text_true_pred)
        else:
            union.update(sample_gt)
            union.update(sample_pred)

        for s in union:
            if s in sample_gt:
                t = s[1]
                gt_flat.append(t2id[t])
                types.add(t)
            else:
                gt_flat.append(0)

            # 是从union中取出来的，所以重复的预测不影响评估结果
            if s in sample_pred:
                t = s[1]
                pred_flat.append(t2id[t])
                types.add(t)
            else:
                pred_flat.append(0)

    labels = [t2id[t] for t in types]
    p, r, f1, support = prfs(gt_flat, pred_flat, labels=labels, average=None, zero_division=1)
    p_micro, r_micro, f1_micro, _ = prfs(gt_flat, pred_flat, labels=labels, average='micro', zero_division=1)
    p_macro, r_macro, f1_macro, _ = prfs(gt_flat, pred_flat, labels=labels, average='macro', zero_division=1)

    return {
        "p": p * 100, "r": r * 100, "f1": f1 * 100, "support": support,
        "p_micro": p_micro * 100, "r_micro": r_micro * 100, "f1_micro": f1_micro * 100,
        "p_macro": p_macro * 100, "r_macro": r_macro * 100, "f1_macro": f1_macro * 100,
        "types": list(types),       # per type里面的类型顺序和这个types一致
    }

@staticmethod
def _print_results(eval_result: Dict[str, Any]):
    per_type = [eval_result["p"], eval_result["r"], eval_result["f1"], eval_result["support"]]
    total_support = sum(eval_result["support"])
    micro = [eval_result["p_micro"], eval_result["r_micro"], eval_result["f1_micro"], total_support]
    macro = [eval_result["p_macro"], eval_result["r_macro"], eval_result["f1_macro"], total_support]
    types: List[str] = eval_result["types"]

    columns = ('type', 'precision', 'recall', 'f1-score', 'support')

    row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
    logger.info(row_fmt % columns)

    metrics_per_type = []
    for i, t in enumerate(types):
        metrics = []
        for j in range(len(per_type)):
            metrics.append(per_type[j][i])
        metrics_per_type.append(metrics)

    def get_row(data, label):
        row = [label]
        for i in range(len(data) - 1):
            row.append("%.2f" % (data[i]))
        row.append(data[3])
        return tuple(row)

    for m, t in zip(metrics_per_type, types):
        logger.info(row_fmt % get_row(m, t))

    logger.info('')
    # micro
    logger.info(row_fmt % get_row(micro, 'micro'))
    # macro
    logger.info(row_fmt % get_row(macro, 'macro'))


def _score_stat(results: List[Dict[str, Any]]):
    t2id = {"None": 0}
    gt_types = set()
    for sample_results in results:
        for gt in sample_results["gts"]:
            t = gt["type"]
            if t not in t2id:
                t2id[t] = len(t2id)
            gt_types.add(t)
        for pred in sample_results["pred"]:
            t = pred["type"]
            if t not in t2id:
                t2id[t] = len(t2id)

    gt, pred = _convert(results, include_entity_types=True)
    ner_score = _score(gt, pred)

    gt_wo_type, pred_wo_type = _convert(results, include_entity_types=False)
    loc_eval_score = _score(gt_wo_type, pred_wo_type)

    cls_eval_score = _score(gt, pred, cls_metric=True)

    logger.info("--- NER ---")
    logger.info("")
    _print_results(ner_score)

    logger.info("")
    logger.info("--- NER on Localization ---")
    logger.info("")
    _print_results(loc_eval_score)

    # logger.info("")
    # logger.info("--- NER on Classification ---")
    # logger.info("")
    # _print_results(cls_eval_score)


@app.command()
def blip2(
    file: str = typer.Argument(
        "data/MEPAVE/jdair.jave.test.uie.jsonl"
    ),
    imgpath: str = typer.Option("data/MEPAVE/product_images"),
    output: str = typer.Option(...),
    blip2_path: str = typer.Option(...),
    lm_path: str = typer.Option("models/chatglm-6b"),
    prompt: str = typer.Option("这件商品的属性是什么？"),
):
    tokenizer, blip2_proc, model = load_blip2chatglm(blip2_path, lm_path)
    device_info = alloc1([])
    device = torch.device(device_info["device"])
    model.to(device)
    model.eval()

    # NOTE: Still slow. I don't know why no performance boost.
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     logger.info("Use torch.compile")
    #     model = torch.compile(model)

    results = []
    output: Path = Path(output)
    output.parent.mkdir(exist_ok=True, parents=True)
    with torch.no_grad(), Progress(*no_total_columns()) as pbar, output.open(
        "w", encoding="utf8"
    ) as wf:
        tid = pbar.add_task("ClipScore", total=None)
        for sample in load_uie_datatset(file, imgpath):
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
            pred = parse_uie(output)
            result = {
                "image": sample["image_path"],
                "label": sample["entities"],
                "output": output,
                "pred": pred
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