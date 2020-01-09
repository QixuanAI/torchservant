# -*- coding: utf-8 -*-
# @Time    : 2020/01/07 23:43
# @Author  : LQX
# @Email   : qxsoftware@163.com
# @File    : weight_initializers.py
# @Software: PyCharm

import torch as t
from torch import nn, save, load, set_grad_enabled
from warnings import warn
from torch.nn import Module, DataParallel
from torch.optim import Optimizer
from torchservant.config import BasicConfig

def initialize_weights(model:Module,config:BasicConfig):
    init_method = config.weight_init_method
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()