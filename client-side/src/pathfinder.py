import typing as t
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class PathConfig:
    base_path: t.Optional[Path] = Path(__file__).absolute().parent.parent
    configs_dir: t.Optional[Path] = None
    data_dir: t.Optional[Path]  = None
    models_dir:t.Optional[Path] = None

    def __post_init__(self):
        self.configs_dir = self.base_path.joinpath("configs")
        self.data_dir = self.base_path.joinpath('data')
        self.models_dir = self.base_path.joinpath('models')

    def to_dict(self):
        paths = asdict(self)
        paths = {k: str(v) for k, v in paths.items()}
        return paths


if __name__ == "__main__":
    path_repo = PathConfig()
    print("Paths" + path_repo.to_dict())