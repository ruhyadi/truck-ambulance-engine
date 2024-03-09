"""Main function."""

import rootutils

ROOT = rootutils.autosetup()

import hydra
from omegaconf import DictConfig

from src.utils.logger import get_logger

log = get_logger()


def detection(cfg: DictConfig) -> None:
    """Detection main function."""
    log.info(f"Start detection...")