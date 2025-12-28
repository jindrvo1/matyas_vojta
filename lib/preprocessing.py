import inspect
from dataclasses import dataclass
from typing import Any, Self, overload

import lib.preprocessing_funcs as funcs
from lib.frame import Frame
from lib.utils import PreprocessorFn, PreprocessorFnWrapper


@dataclass
class Preprocessor:
    steps: list[PreprocessorFnWrapper[PreprocessorFn]]
    Color: funcs.Color = funcs.Color()
    Contrast: funcs.Contrast = funcs.Contrast()
    Enhancement: funcs.Enhancement = funcs.Enhancement()

    def __init__(
        self,
        steps: list[tuple[PreprocessorFn, *tuple[Any, ...]]] | None = None,
    ):
        self.steps = []
        self.add_steps(steps or [])
        self.apply = self.__call__

    def add_step(self, step: PreprocessorFn, *args, **kwargs) -> Self:
        self.steps.append(PreprocessorFnWrapper(step, *args, **kwargs))
        return self

    def add_steps(
        self,
        steps: list[tuple[PreprocessorFn, *tuple[Any, ...]]],
    ) -> Self:
        for step, *step_args in steps:
            arg_names: list[str] = [
                arg_name
                for arg_name in list(inspect.signature(step).parameters.keys())[1:]
            ]
            args = [arg for arg in step_args if not isinstance(arg, dict)]
            kwargs = {
                arg_name: arg_val
                for step_arg in step_args
                if isinstance(step_arg, dict)
                for arg_name, arg_val in step_arg.items()
            }
            bound_kwargs = {}
            for arg_name in arg_names:
                arg_val = kwargs.get(arg_name)
                if arg_val is None and len(args) > 0:
                    arg_val = args.pop(0)
                if arg_val is not None:
                    bound_kwargs[arg_name] = arg_val

            self.add_step(step, **bound_kwargs)

        return self

    @overload
    def __call__(self, frames: Frame) -> Frame: ...
    @overload
    def __call__(self, frames: list[Frame]) -> list[Frame]: ...
    def __call__(self, frames: Frame | list[Frame]) -> Frame | list[Frame]:
        if isinstance(frames, list):
            return [self(frame) for frame in frames]
        elif isinstance(frames, Frame):
            res = frames.copy()
            for step in self.steps:
                res = step(res)
            return res

        raise TypeError(f"Expected Frame or list[Frame], got {type(frames)}")

    def __repr__(self) -> str:
        res = f"{self.__class__.__name__}(\n"
        for i, step in enumerate(self.steps):
            args = ",".join(map(str, step.args))
            kwargs = ",".join(f"{k}={v}" for k, v in step.kwargs.items())
            res += f"\t{step.func.__name__}("
            res += f"{args}{',' if args and kwargs else ''}{f'{kwargs}' if kwargs else ''})"
            res += f"{', ' if i < len(self.steps) - 1 else ''}\n"
        res += ")"

        return res

    def __str__(self) -> str:
        return self.__repr__().replace("\n", "").replace("\t", "")


class PreprocessorFactory:
    Color: funcs.Color = funcs.Color()
    Contrast: funcs.Contrast = funcs.Contrast()
    Enhancement: funcs.Enhancement = funcs.Enhancement()

    defaults: dict[str, list[tuple[PreprocessorFn, *tuple[Any, ...]]]] = {
        "paddle": [
            (Enhancement.denoise_colored,),
            (Contrast.clahe_bgr, 3.0, 8),
            (Enhancement.sharpen,),
        ],
        "easyocr": [
            (Enhancement.denoise_colored,),
            (Color.convert, Color.BGR, Color.GRAY),
            (Contrast.clahe_gray, 3.0, 8),
            (Enhancement.sharpen,),
            (Color.convert, Color.GRAY, Color.BGR),
        ],
    }

    @classmethod
    def for_paddle(cls) -> Preprocessor:
        return cls._construct("paddle")

    @classmethod
    def for_easyocr(cls) -> Preprocessor:
        return cls._construct("easyocr")

    @classmethod
    def _construct(cls, key: str) -> Preprocessor:
        preprocessor = Preprocessor()
        for step in cls.defaults[key]:
            preprocessor.add_step(*step)
        return preprocessor


def preprocess_identity(frame: Frame) -> Frame:
    return frame
