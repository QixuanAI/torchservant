# -*- coding: utf-8 -*-
# @Time    : 2019/09/15 19:34
# @Author  : LQX
# @Email   : qxsoftware@163.com
# @File    : setup.py
# @Software: PyCharm

from setuptools import setup, find_packages

setup(
    name="torchservant",
    version="0.2",
    keywords={"pytorch", "ai", "deeplearning", "framework", "servant"},
    description="Many useful tools for PyTorch.",
    long_description="TorchServant contains the following components:\n" +
                     "\tmodelkeeper:\tManage weights files and checkpoints, keep training process continuous.\n" +
                     "\tboard:\tAPI for visdom and tensorboard, visualize diagrams, illustrations and progresses.\n" +
                     "\tstats\tResource statistics on GPUs, memories, cpu, and consumed time."+
                     "\tclassicalmodels:\tInclude AlexNet, VGG, Resnet, Inception, etc."+
                     "\tobserver:\tVisualize feature maps during training and evaluation process.\n" +
                     "\tweightransfer:\tAn Qt-based visual tool to transfer weights between different models.",
    license="GNU GENERAL PUBLIC LICENSE v3",
    url="https://github.com/QixuanAI/torchservant",
    author="qxsoftware",
    author_email="qxsoftware@163.com",
    packages=find_packages(exclude=["checkpoints", "logs", "test*", "*.md"]),
    install_requires=["torch"],
)
