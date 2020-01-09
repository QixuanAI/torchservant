# -*- coding: utf-8 -*-
# @Time    : 2020/01/07 23:43
# @Author  : LQX
# @Email   : qxsoftware@163.com
# @File    : weight_initializers.py
# @Software: PyCharm

import math
import torch as t
from torch import nn, save, load, set_grad_enabled
from warnings import warn
from torch.nn import Module, DataParallel
from torch.optim import Optimizer
from torchservant.cfgenator.config import BasicConfig

class _BaseWeiInitzer():
    def filldata(self, parameters: t.Tensor):
        raise NotImplementedError

    def __call__(self, parameters: t.Tensor):
        return self.filldata(parameters)


def initialize_weights(model:Module,weight_initzer=None):
    if weight_initzer is None:
        weight_initzer = ZeroWeiInitzer()
    assert weight_initzer is _BaseWeiInitzer or  callable(weight_initzer)
    if weight_initzer is AutoWeiInitzer:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                # weight_initzer(m.weight.data)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    weight_initzer(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                weight_initzer(m.weight.data)
                if m.bias is not None:
                    weight_initzer(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                weight_initzer(m.weight.data)
                weight_initzer(m.bias.data)
            elif isinstance(m, nn.Linear):
                weight_initzer(m.weight.data)
                weight_initzer(m.bias.data)

class ZeroWeiInitzer(_BaseWeiInitzer):
    def filldata(self, parameters: t.Tensor):
        parameters.data.fill_(0)


class OneWeiInitzer(_BaseWeiInitzer):
    def filldata(self, parameters: t.Tensor):
        parameters.data.fill_(1)


class NWeiInitzer(_BaseWeiInitzer):
    def __init__(self, N):
        self.N = N

    def filldata(self, parameters: t.Tensor):
        parameters.data.fill_(self.N)


class XavierWeiInitzer(_BaseWeiInitzer):
    def __init__(self, N):
        raise NotImplementedError

    def filldata(self, parameters: t.Tensor):
        raise NotImplementedError

class AutoWeiInitzer(_BaseWeiInitzer):
    def filldata(self, parameters: t.Tensor):
        raise NotImplementedError

init_dict = {
    0: ZeroWeiInitzer,
    1: OneWeiInitzer,
    'n':NWeiInitzer,
    'auto':AutoWeiInitzer
}
