from typing import List

from ..extention.rich import full_columns, no_total_columns
from . import ItrDataPipeline, LstDataPipeline, DataPipeline
from rich.progress import Progress


@DataPipeline.register()
class ItrToLst(LstDataPipeline):
    def __init__(
        self,
        datapipe: ItrDataPipeline,
        **kwargs
    ):
        super().__init__([])

        with Progress(*no_total_columns()) as pbar:
            tid = pbar.add_task(ItrToLst.__name__, total=None)
            for data in datapipe:
                self.datapipe.append(data)
                pbar.advance(tid)

    def __getitem__(self, index):
        return self.datapipe[index]

    def __len__(self):
        return len(self.datapipe)


@DataPipeline.register()
class SequenceWrapper(LstDataPipeline):
    def __init__(self, datapipe: List, **kwargs):
        super().__init__(datapipe)

    def __getitem__(self, index):
        return self.datapipe[index]

    def __len__(self):
        return len(self.datapipe)
