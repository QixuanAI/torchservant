# -*- coding: utf-8 -*-
# @Time    : 2020/01/10 04:46
# @Author  : LQX
# @Email   : qxsoftware@163.com
# @File    : monitor.py
# @Software: PyCharm


class ProcMonitor():
    backbone='visdom' # choose from visdom or tensorboardx
    port = None

    def __init__(self, backbone='visdom',port=None):
        assert backbone in ['visdom','vis','tensorboardx','tensorboard','tb']
        self.backbone = 'visdom' if backbone in ['vis','visdom'] else 'tensorboardx'
        if isinstance(port,int):
            self.port=port
        elif self.backbone=='visdom':
            self.port=8097
        else:
            self.port=6006