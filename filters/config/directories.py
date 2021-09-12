from pathlib import Path


class _Directories:
    def __init__(self):
        self.project = Path("/app/filters")
        self.project = Path(__file__).parents[1].resolve()
        self.config = self.project.parents[0].resolve() / "config"
        self.output_filename = self.project / "output"
        self.app = self.project.parents[0].resolve()


directories = _Directories()
