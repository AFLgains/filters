# -*- coding: utf-8 -*-
"""A sample CLI."""

import os
import logging

import click
from logging.config import dictConfig

from .utils.config import Config
from .engine.engine import Engine

from .config.directories import directories
from .config.constants import _BASE_CONFIG_FILE

logger = logging.getLogger(__name__)

@click.command()
@click.option(
    "-m",
    "--mode",
    type=click.Choice(["live", "schedule", "train"]),
    default="live",
    show_default=True,
)
@click.option(
    "-e", "--email/--no-email", is_flag=True, default=False, show_default=True
)
@click.option("-l", "--email_list", default=None, show_default=True)
def main(mode: str, email: bool, email_list: str):
    config = Config(directories.config / _BASE_CONFIG_FILE).read()
    dictConfig(config.logging)
    engine = Engine(config, email_list)
    engine.run(mode, email)

def main_for_server():
    config = Config(directories.config / _BASE_CONFIG_FILE).read()
    dictConfig(config.logging)
    engine = Engine(config, [])
    output = engine.run("live", False)
    return output

if __name__ == "__main__":
    main()
