import dataclasses
from datetime import datetime
import json
from pathlib import Path
from transformers.integrations import TensorBoardCallback, WandbCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from src.util.sym import sym_tbl
from src.models.blip2chatglm import *
from src.datasets import DataArguments, blip2chatglm_collator


def main():
    # args
    model_args, data_args, train_args = HfArgumentParser(
        [Blip2ChatGLMModelArguments, DataArguments, TrainingArguments]
    ).parse_args_into_dataclasses()
    timestamp = str(datetime.now()).replace(" ", "_").replace(":", "-")
    train_args.output_dir = f"{train_args.output_dir}/{timestamp}"
    sym_tbl().cfg["model"] = model_args
    sym_tbl().cfg["data"] = data_args
    sym_tbl().cfg["train"] = train_args
    Path(train_args.output_dir).mkdir(parents=True)
    with Path(train_args.output_dir).joinpath("model_args.json").open("w") as wf:
        json.dump(dataclasses.asdict(model_args), wf)
    with Path(train_args.output_dir).joinpath("data_args.json").open("w") as wf:
        json.dump(dataclasses.asdict(data_args), wf)
    with Path(train_args.output_dir).joinpath("train_args.json").open("w") as wf:
        json.dump(dataclasses.asdict(train_args), wf)

    # init model
    model = Blip2ChatGLMTrainer.load_model()

    # load dataset
    if data_args.dataset == "mepave":
        from src.datasets.mepave import MEPAVEDataset

        train_dataset = MEPAVEDataset.load_dataset(
            "train",
            image_size=model.config.vision_config.image_size,
            num_query_tokens=model.config.num_query_tokens,
        )
        dev_dataset = MEPAVEDataset.load_dataset(
            "dev",
            image_size=model.config.vision_config.image_size,
            num_query_tokens=model.config.num_query_tokens,
        )
    elif data_args.dataset == "cocozhvqa":
        from src.datasets.vqa import CoCoZhVQADataset

        train_dataset = CoCoZhVQADataset.load_dataset(
            "train",
            image_size=model.config.vision_config.image_size,
            num_query_tokens=model.config.num_query_tokens,
        )
        dev_dataset = CoCoZhVQADataset.load_dataset(
            "dev",
            image_size=model.config.vision_config.image_size,
            num_query_tokens=model.config.num_query_tokens,
        )
    else:
        raise ValueError(f"Invalid dataset: {data_args.dataset}")

    # start train
    trainer = Blip2ChatGLMTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        args=sym_tbl().cfg["train"],
        data_collator=blip2chatglm_collator,
    )
    trainer.remove_callback(WandbCallback)
    trainer.train()
    # save model (only save peft model)
    model.language_model.save_pretrained(sym_tbl().cfg["train"].output_dir)


if __name__ == "__main__":
    main()
