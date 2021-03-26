from __future__ import absolute_import
from __future__ import print_function

import threading
import numpy as np
import random

def pad_zeros(arr, min_length=None):
    """
    `arr` is an array of `np.array`s

    The function appends zeros to every `np.array` in `arr`
    to equalize their first axis lenghts.
    """
    dtype = arr[0].dtype
    max_len = max([x.shape[0] for x in arr])
    ret = [
        np.concatenate(
            [x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0
        )
        for x in arr
    ]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [
            np.concatenate(
                [x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)],
                axis=0,
            )
            for x in ret
        ]
    return np.array(ret)

class BatchDataGenerator(object):
    def __init__(
        self,
        dataloader,
        batch_size,
        shuffle=False,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle

        self._load_per_patient_data(dataloader)
        self.steps = (len(self.data[1]) + batch_size - 1) // batch_size
        self.lock = threading.Lock()
        self.generator = self._generator()

    def _load_per_patient_data(self, dataloader):
        N = len(dataloader._data["x"])
        x_list = []
        x_last_list = []
        interval_list = []
        mask_list = []
        y_list = []
      
        for i in range(N):
            x = dataloader._data["x"][i]
            y = dataloader._data["y"][i]
            x_last = dataloader._data["last_x"][i]
            mask = dataloader._data["mask"][i]
            interval = dataloader._data["interval"][i]
            interval = (interval - interval.min()) / (interval.max() - interval.min() + 1e-8)
           
            x_list.append(x)
            x_last_list.append(x_last)
            y_list.append(y)
            interval_list.append(interval)
            mask_list.append(mask)
                 
        self.data = [x_list, x_last_list, y_list]
        self.mask = mask_list
        self.interval = interval_list
    
    def _generator(self):
        B = self.batch_size
        while True:
            if self.shuffle:
                # stupid shuffle
                N = len(self.data[1])
                order = list(range(N))
                random.shuffle(order)
                tmp_data = [[None] * N, [None] * N, [None] * N]
                tmp_interval = [None] * N
                tmp_mask = [None] * N
                for i in range(N):
                    tmp_data[0][i] = self.data[0][order[i]]
                    tmp_data[1][i] = self.data[1][order[i]]
                    tmp_data[2][i] = self.data[2][order[i]]
                    tmp_mask[i] = self.mask[order[i]]
                    tmp_interval[i] = self.interval[order[i]]
                self.data = tmp_data
                self.mask = tmp_mask
                self.interval = tmp_interval

            for i in range(0, len(self.data[1]), B):
                x = self.data[0][i : i + B]
                x_last = self.data[1][i : i + B]
                y = self.data[2][i : i + B]
                mask = self.mask[i : i + B]
                interval = self.interval[i : i + B]
                x = pad_zeros(x)  # (B, T, D)
                x_last = pad_zeros(x_last)  # (B, T, D)
                mask = pad_zeros(mask)  # (B, T, D)
                interval = pad_zeros(interval)  # (B, T, D)
                y = np.expand_dims(y, axis=-1)  # (B, T, 1)
                batch_data = (x, x_last, y)
                
                yield {"data": batch_data, "mask": mask, "interval": interval}

    def __iter__(self):
        return self.generator

    def next(self):
        with self.lock:
            return next(self.generator)

    def __next__(self):
        return self.next()