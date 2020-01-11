# -*- coding: utf-8 -*-
# @Time    : 2020/01/10 03:36
# @Author  : LQX
# @Email   : qxsoftware@163.com
# @File    : _basic.py
# @Software: PyCharm

from warnings import warn


def handle_err(msg, errlevel=1):
    if errlevel == 0:
        return True
    elif errlevel == 1:
        warn(msg)
        return True
    else:
        raise RuntimeError(msg)


# del modules not for external reference
