import itertools
import random
from typing import Iterator, List, Optional, Sized
from . import ItrDataPipeline, DataPipeline
from torch.utils.data import get_worker_info


@DataPipeline.register()
class Batch(ItrDataPipeline):

    def __init__(
        self,
        datapipe: ItrDataPipeline,
        batch_size: int = 1,
        drop_last: bool = False,
        **kwargs
    ):
        super().__init__(datapipe)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for sample in iter(self.datapipe):
            batch.append(sample)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            if not self.drop_last:
                yield batch


@DataPipeline.register()
class WithLength(ItrDataPipeline):
    def __init__(self, datapipe: ItrDataPipeline, length: Optional[int], **kwargs):
        super().__init__(datapipe)
        self.length = length

    def __iter__(self) -> Iterator:
        return iter(self.datapipe)

    def __len__(self) -> int:
        if self.length is not None:
            return self.length
        else:
            raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))


@DataPipeline.register()
class Shuffle(ItrDataPipeline):

    def __init__(
        self,
        datapipe: ItrDataPipeline,
        buffer_size: int = 1000,
        **kwargs
    ):
        super().__init__(datapipe)
        self.buffer_size = buffer_size
        self._shuffle_enabled = True

    def set_shuffle_settings(self, shuffle=True):
        self._shuffle_enabled = shuffle

    @staticmethod
    def buffer_replace(buffer, x):
        idx = random.randint(0, len(buffer) - 1)
        val = buffer[idx]
        buffer[idx] = x
        return val

    def __iter__(self) -> Iterator:
        if not self._shuffle_enabled:
            for x in self.datapipe:
                yield x
        else:
            buffer: List = []
            for x in self.datapipe:
                if len(buffer) == self.buffer_size:
                    yield Shuffle.buffer_replace(buffer, x)
                else:
                    buffer.append(x)
            random.shuffle(buffer)
            while buffer:
                yield buffer.pop()


@DataPipeline.register()
class SplitByWorker(ItrDataPipeline):
    def __init__(self, datapipe: ItrDataPipeline, **kwargs):
        super().__init__(datapipe)

    def __iter__(self) -> Iterator:
        worker_info = get_worker_info()
        if worker_info is not None:
            return itertools.islice(self.datapipe, worker_info.id, None, worker_info.num_workers)
        else:
            return iter(self.datapipe)
