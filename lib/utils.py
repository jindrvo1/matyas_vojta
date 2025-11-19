import logging
from functools import wraps
from typing import Callable, TypeAlias, TypedDict

from lib.frame import Frame

PreprocessorFn: TypeAlias = Callable[[Frame], Frame]
Point: TypeAlias = tuple[int, int]


class OCRResult(TypedDict):
    conf: float
    points: list[Point]


BASE_COLORS = [
    (255, 0, 0),  # red
    (0, 255, 0),  # green
    (0, 0, 255),  # blue
    (255, 165, 0),  # orange
    (255, 0, 255),  # magenta
    (0, 255, 255),  # cyan
    (128, 0, 128),  # purple
    (0, 128, 0),  # dark green
]


def suppress_logging(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        level = logging.ERROR

        previous_levels = {
            name: logging.getLogger(name).level
            for name in logging.root.manager.loggerDict
        }
        previous_root = logging.getLogger().level

        try:
            logging.getLogger().setLevel(level)
            for name in previous_levels:
                logging.getLogger(name).setLevel(level)

            return func(*args, **kwargs)

        finally:
            logging.getLogger().setLevel(previous_root)
            for name, prev_level in previous_levels.items():
                logging.getLogger(name).setLevel(prev_level)

    return wrapper
