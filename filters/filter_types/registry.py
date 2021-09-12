from dataclasses import dataclass
from typing import List, Callable


@dataclass
class FilterRecord:
    name: str
    func: Callable
    resources: List[str]  # same with resources


class FilterRegistry:
    def __init__(self):
        self._registry = {}

    def __iter__(self):
        return (filter for filter in self._registry.keys())

    @property
    def registry(self):
        return self._registry

    def get(self, name):
        return self._registry[name]

    def register(self, obj, name, *, resources = None):
        resources = resources or []
        self._registry[name] = FilterRecord(
            name=name, func=obj, resources=resources
        )
