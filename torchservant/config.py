# -*- coding: utf-8 -*-
# @Time    : 2019/1/6 4:23
# @Author  : LQX
# @Email   : qxsoftware@163.com
# @File    : config.py
# @Software: PyCharm

import os
from warnings import warn
from time import strftime as timestr


class BasicConfig(object):
    # data config
    train_data = None  # a path to training data
    val_data = None  # a path to validation data
    classes_list = []
    shuffle_train = True
    shuffle_val = False
    drop_last_train = True
    drop_last_val = False

    # S/L config
    weight_load_path = r'checkpoints/pretrain.pth'  # where to load pre-trained weight for further training
    weight_save_path = r'checkpoints/myweight{time}.pth'  # where to save trained weights for further usage
    log_root = r'logs'  # where to save logs, includes temporary weights of model and optimizer, train_record json list
    history_root = r'history'
    # debug_flag_file = r'debug'

    # efficiency config
    use_gpu = True  # if there's no cuda-available GPUs, this will turn to False automatically
    num_data_workers = 1  # how many subprocesses to use for data loading
    pin_memory = False  # only set to True when your machine's memory is large enough
    time_out = 0  # max seconds for loading a batch of data, 0 means non-limit
    max_epoch = 100  # how many epochs for training
    batch_size = 20  # how many scene images for a batch
    ckpt_freq = 20  # save checkpoint after these iterations
    # ToDo:
    reproducible_record = False  # If set to True, detailed information will be recorded in every iteration for entirely reproducing

    # model config
    model = None
    mode = "train"  # optional from 'train', 'val' or 'eval'
    use_batch_norm = True
    # loss_type = None
    # optimizer_type = None

    # visualize config
    visual_engine = 'visdom'
    host = 'localhost'
    port = None
    visdom_env = 'main'

    def __init__(self, **kwargs):
        self.init_time = timestr('%Y%m%d.%H%M%S')
        if self.mode not in ["train", "training", "inference", "validation", "val", "test", "evaluation", "eval"]:
            warn("Invalid argument mode, expect 'train' or 'inference' but got '%s'" % self.mode)
        self.enable_grad = self.mode in ['train', "training"]
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warn("{} has no attribute {}:{}".format(type(self), key, value))

        # data config
        self.num_classes = len(self.classes_list)

        # efficiency config
        if self.use_gpu:
            from torch.cuda import is_available as cuda_available, device_count
            if cuda_available():
                self.num_gpu = device_count()
                self.gpu_list = list(range(self.num_gpu))
                assert self.batch_size % self.num_gpu == 0, \
                    "Can't split a batch of data with batch_size {} averagely into {} gpu(s)" \
                        .format(self.batch_size, self.num_gpu)
            else:
                warn("Can't find available cuda devices, use_gpu will be automatically set to False.")
                self.use_gpu = False
                self.num_gpu = 0
                self.gpu_list = []
        else:
            from torch.cuda import is_available as cuda_available
            if cuda_available():
                warn("Available cuda devices were found, please switch use_gpu to True for acceleration.")
            self.num_gpu = 0
            self.gpu_list = []
        if self.use_gpu:
            self.map_location = lambda storage, loc: storage
        else:
            self.map_location = "cpu"

        # weight S/L config
        self.vis_env_path = os.path.join(self.log_root, 'visdom')
        os.makedirs(os.path.dirname(self.weight_save_path), exist_ok=True)
        os.makedirs(self.log_root, exist_ok=True)
        os.makedirs(self.vis_env_path, exist_ok=True)
        assert os.path.isdir(self.log_root)
        self.temp_ckpt_path = os.path.join(self.log_root, 'ckpt-{time}.pth'.format(time=self.init_time))
        # self.temp_optim_path = os.path.join(self.log_root, 'tmpoptim{time}.pth'.format(time=self.init_time))
        self.log_file = os.path.join(self.log_root, '{}.{}.log'.format(self.mode, self.init_time))
        self.val_result = os.path.join(self.log_root, 'validation_result{}.txt'.format(self.init_time))
        self.train_record_file = os.path.join(self.log_root, 'train.record.jsons')
        """
       record training process by core.make_checkpoint() with corresponding arguments of
       [epoch, start time, elapsed time, loss value, train accuracy, validate accuracy]
       DO NOT CHANGE IT unless you know what you're doing!!!
       """
        self.__record_fields__ = ['epoch', 'start', 'elapsed', 'loss', 'train_score', 'val_score']
        if len(self.__record_fields__) == 0:
            warn(
                '{}.__record_fields__ is empty, this may cause unknown issues when save checkpoint into {}' \
                    .format(type(self), self.train_record_file))
            self.__record_dict__ = '{{}}'
        else:
            self.__record_dict__ = '{{'
            for field in self.__record_fields__:
                self.__record_dict__ += '"{}":"{{}}",'.format(field)
            self.__record_dict__ = self.__record_dict__[:-1] + '}}'

        # visualize config
        if self.visual_engine in ["visdom", "vis"]:
            self.port = 8097 if self.port is None else self.port
        elif self.visual_engine in ["tensorboardx", "tensorboard", "tb"]:
            self.port = 6006 if self.port is None else self.port
        else:
            raise RuntimeError("Invalid parameter value of visual_engine :", self.visual_engine)

    def __str__(self):
        """:return Configuration details."""
        str = "Configurations for %s:\n" % self.mode
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                str += "{:30} {}\n".format(a, getattr(self, a))
        return str

    def __format_path__(self, path, epoch=None, loss=None, score=None, optim=None):
        model = self.model
        if isinstance(model, function):
            model = model.__name__
        if isinstance(loss, function):
            loss = loss.__name__
        if isinstance(optim, function):
            optim = optim.__name__
        return path.format(time=self.init_time, mode=self.mode, epoch=epoch, model=model, loss=loss,
                           score=score, optim=optim)
