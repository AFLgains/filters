# -*- coding: utf-8 -*-
"""A sample CLI."""

import click
from logging.config import dictConfig

from filters.utils.config import Config
from filters.engine.engine import Engine


@click.command()
@click.option(
    "--config",
    default="config/main.yml",
    show_default=True,
    help="configuration file",
)
@click.version_option(version="0.1.12")
def main(config: str):
    # init the config from config/
    config = Config(config).read()
    # init logging package
    dictConfig(config.logging)

    # init sample code
    engine = Engine(config)
    engine.run(config.mode)


if __name__ == "__main__":
    main()
