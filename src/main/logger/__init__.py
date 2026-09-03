from __future__ import annotations

import sys

import loguru
from loguru import logger

from src import ROOT_PATH


_logger: loguru.Logger | None = None


def configure(level: str = "INFO"):
    global _logger
    if _logger is None:
        _logger = logger
    _logger.remove()
    # rotation/retention 防止日志无限增长
    _logger.add(
        ROOT_PATH + "/output/logs/debug.log",
        level="DEBUG",
        rotation="50 MB",
        retention="30 days",
    )
    _logger.add(
        ROOT_PATH + "/output/logs/info.log",
        level="INFO",
        rotation="50 MB",
        retention="30 days",
    )
    _logger.add(
        ROOT_PATH + "/output/logs/warn.log",
        level="WARNING",
        rotation="50 MB",
        retention="30 days",
    )
    _logger.add(
        ROOT_PATH + "/output/logs/error.log",
        level="ERROR",
        rotation="50 MB",
        retention="30 days",
    )
    _logger.add(sys.stdout, level=level)


def get_logger(level: str = "INFO") -> loguru.Logger:
    global _logger
    if _logger is None:
        configure(level)
    if _logger is None:
        raise RuntimeError("Logger has not been configured. Call configure() first.")
    return _logger


LOG = get_logger()
