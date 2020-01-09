# -*- coding: utf-8 -*-
# @Time    : 2019/09/15 20:44
# @Author  : LQX
# @Email   : qxsoftware@163.com
# @File    : dataset.py
# @Software: PyCharm


from torch.utils.data import Dataset, DataLoader
from cfgenator.config import BasicConfig


def GetDataLoader(data_type, dataset, config):
    # type:(str, Dataset, BasicConfig) -> DataLoader
    if data_type in ["train", "training"]:
        shuffle = config.shuffle_train
        drop_last = config.drop_last_train
    elif data_type in ["val", "validation", "inference"]:
        shuffle = config.shuffle_val
        drop_last = config.drop_last_val
    else:
        raise RuntimeError("Invalid argument 'data_type', expected 'train' or 'val'.")
    assert len(dataset) > config.batch_size
    return DataLoader(dataset, config.batch_size, shuffle, num_workers=config.num_data_workers,
                      pin_memory=config.pin_memory, drop_last=drop_last, timeout=config.time_out)

