import logging
from functools import wraps
from typing import (
    Any,
    Callable,
    Concatenate,
    Generic,
    ParamSpec,
    TypeAlias,
    TypedDict,
    TypeVar,
)

from lib.frame import Frame

Point: TypeAlias = tuple[int, int]

P = ParamSpec("P")
PreprocessorFn: TypeAlias = Callable[Concatenate[Frame, P], Frame]
PreprocessorFnTypeVar = TypeVar("PreprocessorFnTypeVar", bound=PreprocessorFn)


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


class PreprocessorFnWrapper(Generic[PreprocessorFnTypeVar]):
    func: PreprocessorFnTypeVar
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __init__(self, func: PreprocessorFnTypeVar, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, frame: Frame) -> Any:
        return self.func(frame, *self.args, **self.kwargs)

    def __repr__(self) -> str:
        res = f"{self.__class__.__name__}(\n"
        res += f"\t'func': {self.func.__name__}, \n"
        res += f"\t'args': {self.args}, \n"
        res += f"\t'kwargs': {self.kwargs}\n"
        res += ")"
        return res

    def __str__(self) -> str:
        return self.__repr__().replace("\n", "").replace("\t", "")


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
