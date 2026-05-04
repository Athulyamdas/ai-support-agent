"""
app/utils/logger.py — Colourised structured logger used across the project.
"""

import logging
import colorlog

_FMT = "%(log_color)s%(levelname)-8s%(reset)s | %(cyan)s%(name)s%(reset)s | %(message)s"

_COLOR_MAP = {
    "DEBUG": "white",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}


def get_logger(name: str) -> logging.Logger:
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(_FMT, log_colors=_COLOR_MAP))

    logger = logging.getLogger(name)
    if not logger.handlers:          # avoid duplicate handlers on re-import
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger