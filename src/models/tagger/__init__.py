from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, Union
from loguru import logger

import torch
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss, KLDivLoss, Embedding
import torch.nn.functional as F
from transformers import BertConfig, RobertaConfig, PreTrainedModel, AutoConfig, AutoModel

from alchemy import AlchemyModel, sym_tbl
from alchemy.pipeline import OutputPipeline
from alchemy.util import batch_to_device, filter_optional_cfg
from alchemy.util.optim import prepare_trf_based_model_params
from ...task.ner.tagging import BioTaggingScheme, IOTaggingScheme, TaggingScheme
from ...task.ner import NerTask
from .model import Tagger


@OutputPipeline.register()
class ProcTaggingOutput(OutputPipeline):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, outputs: Union[Dict[str, Any], List], inputs: MutableMapping[str, Any]) -> Union[Dict[str, Any], List]:
        ret = []
        scheme: TaggingScheme = sym_tbl().model.tagging_scheme

        tags = outputs["tags"]
        token_masks = inputs['token_masks']
        token_count = torch.sum(token_masks.long(), dim=-1).cpu().numpy()

        for sample_i, (pred_tag, tok_cnt) in enumerate(zip(tags, token_count)):
            preds = scheme.decode_tags(pred_tag[: tok_cnt])

            sample_outputs = []
            for pred_left, pred_right, pred_type in preds:
                # NOTE: 注意decoding tags得到的是range而不是span，右边界是exclusive的
                sample_outputs.append(
                    {"start": pred_left, "end": pred_right - 1, "type": pred_type}
                )
            ret.append(sample_outputs)
        return ret


@AlchemyModel.register("Tagger")
class AlchemyTagger(AlchemyModel):
    MODEL_CLASS = Tagger

    @classmethod
    def save(cls, model: Module, path: Path, **kwargs):
        path.mkdir(parents=True, exist_ok=True)
        if isinstance(model, PreTrainedModel):
            model.save_pretrained(path)
        else:
            raise ValueError("Unimplemented model type \"{}\"".format(type(model)))

    def __init__(self):
        super().__init__()
        task: NerTask = sym_tbl().task
        # scheme和模型绑定而不是和task绑定，所以应该放在模型里
        self.tagging_scheme = BioTaggingScheme(task.entity_types)

        if "model_path" in self.model_cfg:
            self.config = AutoConfig.from_pretrained(self.model_cfg["model_path"])
            model = self.MODEL_CLASS.from_pretrained(
                self.model_cfg["model_path"],
                config=self.config,
                num_tags=self.tagging_scheme.num_tags,

                # 一些可选的config，即如果文件中没有配置，那么就默认
                **filter_optional_cfg(self.model_cfg, {
                    "dropout",
                })
            )
        else:
            # create model
            self.config = AutoConfig.from_pretrained(self.model_cfg["plm_path"])
            model = self.MODEL_CLASS(
                config=self.config,
                num_tags=self.tagging_scheme.num_tags,

                # 一些可选的config，即如果文件中没有配置，那么就默认
                **filter_optional_cfg(self.model_cfg, {
                    "dropout",
                })
            )

            model.encoder = AutoModel.from_pretrained(self.model_cfg["plm_path"])
        self.model = model

    def to(self, *args, **kwargs):
        ret = super().to(*args, **kwargs)
        # if torch.__version__ >= "2.0":
        #     logger.info("Use torch.compile")
        #     self.model = torch.compile(ret)
        return self.model

    def max_positions(self):
        """和Bert一样

        Returns:
            _type_: _description_
        """
        return self.config.max_position_embeddings

    def set_requires_grad(self, requires_grad: bool, mode: str, **kwargs):
        if mode == "encoder":
            for _, param in self.module.encoder.named_parameters():
                param.requires_grad = requires_grad
        else:
            raise NotImplementedError(mode)

    def optim_params(self, **kwargs):
        return prepare_trf_based_model_params(
            self.model,
            self.module.encoder,
            **filter_optional_cfg(kwargs, {"weight_decay", "trf_lr"}),
        )

    def forward(
        self,
        batch: MutableMapping[str, Any],
        needs_loss: bool,
        requires_grad: bool,
        **kwargs
    ) -> MutableMapping[str, Any]:
        batch = batch_to_device(batch, sym_tbl().device)

        hidden, logits = self.model.forward(
            encodings=batch["encoding"],
            encoding_masks=batch["encoding_masks"],
            token2start=batch["token2start"],
        )
        token_masks = batch['token_masks']

        outputs = {
            "hidden": hidden,
            "logits": logits,
            "tags": torch.argmax(logits, dim=-1).detach().cpu().numpy()
        }

        if needs_loss:
            bp_masks = token_masks.flatten()     # bsz * n_tokens

            gts = batch["gt_seq_labels"]
            flat_gts = gts.flatten()[bp_masks]
            flat_logits = logits.flatten(0, 1)[bp_masks]

            # 负采样
            neg_ratio = self.criterion_cfg.get("neg_ratio", -1)
            if neg_ratio < 0:
                pass
            else:
                # 做一些随机负采样
                sample_masks = (flat_gts != 0) | (torch.rand_like(flat_gts, dtype=torch.float) < neg_ratio)
                flat_logits = torch.masked_select(flat_logits, sample_masks.unsqueeze(-1)).view(-1, logits.size(-1))
                flat_gts = torch.masked_select(flat_gts, sample_masks)

            if self.criterion_cfg["type"] == "ce":
                loss_fct = CrossEntropyLoss()
            elif self.criterion_cfg["type"] == "gce":
                gce_q = self.criterion_cfg.get("q")
                raise NotImplementedError()
            else:
                raise NotImplementedError(f"Unknown criterion {self.criterion_cfg}")
            loss = loss_fct.forward(flat_logits, flat_gts)

            outputs["loss"] = self.backward.backward(loss, requires_grad=requires_grad)
        return outputs


# @AlchemyModel.register("CRFTagger")
# class AlchemyCRFTagger(AlchemyTagger):
#     from .crf import CRFTagger
#
#     MODEL_CLASS = CRFTagger
#
#     def forward(
#         self,
#         batch: MutableMapping[str, Any],
#         needs_loss: bool,
#         requires_grad: bool,
#         **kwargs
#     ) -> MutableMapping[str, Any]:
#         batch = batch_to_device(batch, sym_tbl().device)
#
#         hidden, emissions = self.model.forward(
#             encodings=batch["encoding"],
#             encoding_masks=batch["encoding_masks"],
#             token2start=batch["token2start"],
#         )
#
#         token_masks = batch['token_masks']
#
#         tags = self.model.crf.decode(emissions, mask=token_masks)
#         outputs = {
#             "hidden": hidden,
#             "emissions":emissions,
#             "tags": tags,
#         }
#
#         if needs_loss:
#             loss = -self.model.crf.forward(
#                 emissions, batch["gt_seq_labels"], mask=token_masks,
#                 reduction="token_mean",
#             )
#
#             outputs["loss"] = self.backward.backward(loss, requires_grad=requires_grad)
#         return outputs
#