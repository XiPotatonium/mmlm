import dataclasses
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import logging
import datasets
import transformers
from transformers.integrations import TensorBoardCallback, WandbCallback, TrainerCallback, ProgressCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from alchemy import sym_tbl
from src.models.blip2chatglm import *
from src.pipeline.lm import DataArguments, blip2chatglm_collator


logger = logging.getLogger(__name__)


class MyProgressCallback(ProgressCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        super().on_log(args, state, control, logs, **kwargs)
        if state.is_local_process_zero:
            logger.info(str(logs))


def main():
    # args
    model_args, data_args, train_args = HfArgumentParser(
        [Blip2ChatGLMModelArguments, DataArguments, TrainingArguments]
    ).parse_args_into_dataclasses()
    timestamp = str(datetime.now()).replace(" ", "_").replace(":", "-")
    train_args.output_dir = f"{train_args.output_dir}/{timestamp}"
    train_args.logging_dir = train_args.output_dir
    Path(train_args.output_dir).mkdir(parents=True)

    # setup logging
    log_level = train_args.get_process_log_level()
    print(f"Log level = {log_level}")
    logger.setLevel(log_level)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    file_handler = logging.FileHandler(
        os.path.join(train_args.output_dir, f"log.txt")
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    # prepare configs
    sym_tbl().cfg["model"] = model_args
    sym_tbl().cfg["data"] = data_args
    sym_tbl().cfg["train"] = train_args
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
        from src.pipeline.lm.mepave import MEPAVEDataset

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
        from src.pipeline.lm.vqa import CoCoZhVQADataset

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
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(MyProgressCallback)
    train_result = trainer.train()
    print(train_result)
    # save model (only save peft model)
    # model.language_model.save_pretrained(sym_tbl().cfg["train"].output_dir)


if __name__ == "__main__":
    main()
