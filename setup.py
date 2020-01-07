# -*- coding: utf-8 -*-
# @Time    : 2019/09/15 19:34
# @Author  : LQX
# @Email   : qxsoftware@163.com
# @File    : setup.py
# @Software: PyCharm

from setuptools import setup, find_packages

with open("README_en-US.md", "r") as fh:
    long_description = fh.read()

setup(
    name="torchservant",
    version="0.2.0",
    keywords={"pytorch", "ai", "deeplearning", "framework", "servant"},
    description="Many useful tools for PyTorch.",
    long_description=long_description,
    license="GNU GENERAL PUBLIC LICENSE v3",
    url="https://github.com/QixuanAI/torchservant",
    author="qxsoftware",
    author_email="qxsoftware@163.com",
    packages=find_packages(exclude=["checkpoints", "logs", "test*", "*.md"]),
    install_requires=["torch"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
)
