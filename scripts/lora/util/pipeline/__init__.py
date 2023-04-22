# __future__.annotations will become the default in Python 3.11
from __future__ import annotations

from typing import Any, Dict, Iterator, MutableMapping, Optional, Union, List
from abc import ABC, abstractmethod
from torch.utils.data import IterableDataset, Dataset
from loguru import logger

from ..registry import Registrable


class DataPipeline(Registrable):
    """Note that DataPipeline might be executed in a different process.
    Therefore be careful in setting fields, especially large objects
    """
    @classmethod
    def from_registry(
        cls,
        ty: str,
        datapipe: Optional[Iterator] = None,
        **kwargs
    ):
        pipeline_cls = cls.resolve_registered_module(ty)
        try:
            return pipeline_cls(datapipe, **kwargs)
        except Exception as e:
            logger.error("Error initializing {}".format(pipeline_cls))
            raise e

    def __init__(self, datapipe) -> None:
        super().__init__()
        self.datapipe = datapipe


class ItrDataPipeline(DataPipeline, IterableDataset):

    def __init__(self, datapipe) -> None:
        super().__init__(datapipe)

    def __iter__(self) -> Iterator:
        raise NotImplementedError()


class LstDataPipeline(DataPipeline, Dataset):

    def __init__(self, datapipe) -> None:
        super().__init__(datapipe)

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class OutputPipeline(Registrable):
    """OutputPipeline和DataPipeline不同的是，它返回的不是Iterator，而是input的处理结果，__call__可以理解为一个mapper

    Args:
        Registrable (_type_): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    @classmethod
    def from_registry(
        cls,
        ty: str,
        **kwargs
    ):
        pipeline_cls = cls.resolve_registered_module(ty)
        try:
            return pipeline_cls(**kwargs)
        except Exception as e:
            logger.error("Error initializing {}".format(pipeline_cls))
            raise e

    def __call__(self, outputs: Union[Dict[str, Any], List], inputs: MutableMapping[str, Any]) -> Union[Dict[str, Any], List]:
        raise NotImplementedError()


class EvalPipeline(Registrable):
    @classmethod
    def from_registry(cls, ty: str, **kwargs):
        pipeline_cls = cls.resolve_registered_module(ty)
        try:
            return pipeline_cls(**kwargs)
        except Exception as e:
            logger.error("Error initializing {}".format(pipeline_cls))
            raise e

    def begin_eval(self, split: str, **kwargs):
        pass

    def __call__(self, split: str, **kwargs) -> Dict[str, Any]:
        pass


class SchedPipeline(ABC, Registrable):
    @classmethod
    def from_registry(cls, ty: str, **kwargs):
        cb_cls = cls.resolve_registered_module(ty)
        if not issubclass(cb_cls, (
            BeginEpochPipeline, EndEpochPipeline, BeginStepPipeline, EndStepPipeline
        )):
            raise ValueError(
                "Callback ({}: {}) must extend {} or {} or {} or {}".format(
                    ty,
                    cb_cls.__name__,
                    BeginEpochPipeline.__name__,
                    EndEpochPipeline.__name__,
                    BeginStepPipeline.__name__,
                    EndStepPipeline.__name__,
                )
            )

        try:
            return cb_cls(**kwargs)
        except Exception as e:
            logger.error("Error initializing {}".format(cb_cls))
            raise e

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, **kwargs) -> Dict[str, Any]:
        return kwargs


class BeginEpochPipeline(SchedPipeline):
    def __init__(self) -> None:
        super().__init__()

class EndEpochPipeline(SchedPipeline):
    def __init__(self) -> None:
        super().__init__()

class BeginStepPipeline(SchedPipeline):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, batch: MutableMapping[str, Any], **kwargs) -> Dict[str, Any]:
        return kwargs

class EndStepPipeline(SchedPipeline):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, outputs: MutableMapping[str, Any], **kwargs) -> Dict[str, Any]:
        return kwargs
