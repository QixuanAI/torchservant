# -*- coding: utf-8 -*-
# @Time    : 2019/09/16 01:33
# @Author  : LQX
# @Email   : qxsoftware@163.com
# @File    : tensorboardx_api.py
# @Software: PyCharm

import tensorboardX as tb
from torchservant.cfgenator.config import BasicConfig

class TensorBoardX():
    def __init__(self,config:BasicConfig):
        self.writer = tb.SummaryWriter(logdir=config.log_root+"_tbx",comment=config.log_file)

    def plot(self, y, x, line_name, win, legend=None):
        self.writer.add_scalar(win,y,x)
