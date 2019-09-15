# -*- coding: utf-8 -*-
# @Time    : 2019/09/15 19:34
# @Author  : LQX
# @Email   : qxsoftware@163.com
# @File    : setup.py
# @Software: PyCharm

from setuptools import setup, find_packages

setup(
    name="torch_engine",
    version="0.1",
    keywords={"pytorch", "ai", "deeplearning", "framework", "engine"},
    description="A Framework for developing Deep Learning Neural Networks with PyTorch",
    license="GNU GENERAL PUBLIC LICENSE v3",
    url="https://github.com/QixuanAI/pytorch_AI_Engine",
    author="qxsoftware",
    author_email="qxsoftware@163.com",
    packages=find_packages(exclude=["checkpoints", "logs", "test*", "*.md"]),
    install_requires=["torch"],
)
