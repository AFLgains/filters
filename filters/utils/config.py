# -*- coding: utf-8 -*-
"""
Config manager
"""

import os
import yaml
from box import Box

class Loader(yaml.FullLoader):
    """https://stackoverflow.com/questions/528281/how-can-i-include-a-yaml-file-inside-another"""

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]
        super(Loader, self).__init__(stream)

    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, 'r') as f:
            return yaml.load(f, Loader)

Loader.add_constructor('!include', Loader.include)

class Config:
    def __init__(self, config_path: str):
        self._config_path = config_path

    def read(self) -> Box:
        """Reads main config file"""
        if os.path.isfile(self._config_path) and os.access(self._config_path, os.R_OK):
            with open(str(self._config_path),'r') as f:
                config = yaml.load(f, Loader=Loader)
            return Box(config)
        else:
            raise FileNotFoundError(self._config_path)
