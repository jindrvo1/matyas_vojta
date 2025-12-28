import cv2

from lib.frame import Frame


class Color:
    RGB: str = "RGB"
    BGR: str = "BGR"
    GRAY: str = "GRAY"

    _channels_mapping: dict[tuple[str, str], int] = {
        (RGB, BGR): cv2.COLOR_RGB2BGR,
        (RGB, GRAY): cv2.COLOR_RGB2GRAY,
        (BGR, RGB): cv2.COLOR_BGR2RGB,
        (BGR, GRAY): cv2.COLOR_BGR2GRAY,
        (GRAY, RGB): cv2.COLOR_GRAY2RGB,
        (GRAY, BGR): cv2.COLOR_GRAY2BGR,
    }

    @classmethod
    def convert(cls, frame: Frame, channel_from: str, channel_to: str) -> Frame:
        res = cv2.cvtColor(frame, code=cls._get_cv2_code(channel_from, channel_to))

        return Frame(res)

    @classmethod
    def _get_cv2_code(cls, channel_from: str, channel_to: str) -> int:
        mapping = cls._channels_mapping.get((channel_from, channel_to))
        if mapping is None:
            raise ValueError(
                f"Unsupported conversion from {channel_from} to {channel_to}"
            )

        return mapping
