import logging

logger = logging.getLogger(__name__)


class FiltersGenerator:

    def __init__(self, registry):
        self.registry = registry
        self._state = {}

    def generate(self, stock_price_list):
        for filter_name in self.registry:
            logger.debug(f"Generating filter '{filter_name}'...")
            record = self.registry.get(filter_name)
            resources =  record.resources
            self._state[filter_name] = record.func(stock_price = stock_price_list,name = filter_name, **resources)

        logger.debug("Filter generation done!")
        return list(self._state.values())