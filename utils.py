import logging
import random
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setLogger(logfile=None):
    logger = logging.getLogger("vcpip")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    logger.addHandler(logging.StreamHandler())
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        logger.addHandler(logging.FileHandler(logfile, encoding="utf-8"))
    for handler in logger.handlers:
        handler.setFormatter(logging.Formatter("%(message)s"))
    return logger
