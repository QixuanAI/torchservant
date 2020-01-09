# -*- coding: utf-8 -*-
# @Time    : 2020/01/10 02:53
# @Author  : LQX
# @Email   : qxsoftware@163.com
# @File    : getcudamodel.py
# @Software: PyCharm

import os
from torch import load, set_grad_enabled
from torch.nn import Module, DataParallel
from torchservant._basic import *
from torchservant.cfgenator.config import BasicConfig


def pack_model(model: Module, use_gpu=True, weight_load_path="", gpu_list=None, printproc=True, errlevel=1) -> Module:
    """
    Try to pack the model into GPU devices and load pre-trained weights.
    :param model: An instance of torch.Module or any inherited class.
    :param config: An instance of core.BasicConfig or any inherited class.
    :return: The packed model.
    """
    # move core to GPU
    if use_gpu:
        try:
            model = model.cuda(0)
        except Exception as e:
            handle_err("Error moving to CUDA because {}".format(e), errlevel)
        map_location = lambda storage, loc: storage
    else:
        map_location = "cpu"
    # load weights
    if os.path.isfile(weight_load_path):
        try:
            state_dict = load(weight_load_path, map_location=map_location)
            model.load_state_dict(state_dict)
            if printproc:
                print('Loaded weights from ' + weight_load_path)
        except Exception as e:
            handle_err('Failed to load weights file {} because {}'.format(weight_load_path, e), errlevel)
    elif printproc and errlevel > 0:
        print('File not exist in {}'.format(weight_load_path))
    # parallel processing
    if use_gpu:
        try:
            if isinstance(gpu_list, list):
                model = DataParallel(model, gpu_list)
            else:
                model = DataParallel(model)
        except Exception as e:
            handle_err('Failed to pack model into parallel because {}'.format(e), errlevel)
    return model


def get_model(model_name, config: BasicConfig, **kwargs) -> Module:
    """
    Find a Module specified by model_name from config.model_libs, and get an instance of it.
    :param config: An instance of core.BasicConfig or any inherited class.
    :param kwargs: Arguments that will be passed into the Module, default is None.
    :return: An instance of torch.Module specified by config.model.
    """
    assert isinstance(config, BasicConfig)
    for lib in config.model_libs[::-1]:
        if hasattr(lib, model_name):
            try:
                with set_grad_enabled(config.enable_grad):
                    model = getattr(lib, model_name)(config, **kwargs)
                    model = pack_model(model, config.use_gpu, config.weight_load_path, config.gpu_list,
                                       config.printproc, config.errlevel)
                    return model
            except Exception as e:
                handle_err(e, config.errlevel)
    handle_err("Can't find {} in {}".format(model_name, config.model_libs), config.errlevel)
    return Module()
