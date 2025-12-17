import random

import hydra
import numpy as np
import torch
from easydict import EasyDict
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from typing import Tuple

from train_animal10n import ANIMAL_Trainer
from train_cifar import CIFAR_Trainer
from train_cifarN import CIFARN_Trainer
from train_red import RED_Trainer


TRAINER_REGISTRY = {
    "cifar": CIFAR_Trainer,
    "cifarn": CIFARN_Trainer,
    "red": RED_Trainer,
    "animal": ANIMAL_Trainer,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prepare_config(cfg: DictConfig) -> Tuple[EasyDict, str]:
    exp_cfg = OmegaConf.to_container(cfg.experiment, resolve=True)
    if exp_cfg is None:
        raise ValueError("Missing experiment configuration.")

    trainer_name = exp_cfg.pop("trainer", None)
    if trainer_name is None:
        raise ValueError("'trainer' must be specified inside the experiment config.")

    if "root_dir" in exp_cfg and exp_cfg["root_dir"]:
        exp_cfg["root_dir"] = to_absolute_path(str(exp_cfg["root_dir"]))

    return EasyDict(exp_cfg), trainer_name


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(f"Optimize in {cfg.optim_goal}")
    seed_everything(cfg.seed)

    config, trainer_key = prepare_config(cfg)
    trainer_cls = TRAINER_REGISTRY.get(trainer_key)
    if trainer_cls is None:
        raise ValueError(f"Trainer '{trainer_key}' is not registered.")

    trainer = trainer_cls(config, cfg.desc)
    trainer.pipeline(trainer.train)


if __name__ == "__main__":
    main()
