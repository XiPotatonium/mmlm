from datetime import datetime
from transformers.integrations import TensorBoardCallback, WandbCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from src.util.sym import sym_tbl
from src.models.blip2chatglm import *


def main():
    # args
    model_args, data_args, train_args = HfArgumentParser(
        [Blip2ChatGLMModelArguments, Blip2ChatGLMDataArguments, TrainingArguments]
    ).parse_args_into_dataclasses()
    timestamp = str(datetime.now()).replace(' ', '_').replace(':', '-')
    train_args.output_dir = f"{train_args.output_dir}/{timestamp}"
    sym_tbl().cfg["model"] = model_args
    sym_tbl().cfg["data"] = data_args
    sym_tbl().cfg["train"] = train_args

    # init model
    model = Blip2ChatGLMTrainer.load_model()

    # load dataset
    train_dataset = MEPAVEDataset.load_dataset("train")
    dev_dataset = MEPAVEDataset.load_dataset("dev")

    # start train
    trainer = Blip2ChatGLMTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        args=sym_tbl().cfg["train"],
        data_collator=mepave_collator,
    )
    trainer.remove_callback(WandbCallback)
    trainer.train()
    # save model (only save peft model)
    model.language.save_pretrained(sym_tbl().cfg["train"].output_dir)


if __name__ == "__main__":
    main()
