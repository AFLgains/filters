# -*- coding: utf-8 -*-
from distutils.core import setup

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="filters",
    version="0.1dev",
    packages=[
        "filters",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=required,
)
