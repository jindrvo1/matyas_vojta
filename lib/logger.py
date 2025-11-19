import logging


class Logger:
    _loggers: dict = {}
    _level: int = logging.INFO

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        if name in Logger._loggers:
            return Logger._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(Logger._level)

        handler = logging.StreamHandler()
        handler.setLevel(logging.NOTSET)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] [%(name)s.%(funcName)s]: %(message)s",
                "%H:%M:%S",
            )
        )

        logger.addHandler(handler)
        Logger._loggers[name] = logger
        return logger

    @staticmethod
    def set_level(level: int | str):
        str_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        level = level if isinstance(level, int) else str_map[level]
        Logger._level = level

        for logger in Logger._loggers.values():
            logger.setLevel(Logger._level)


class Loggable:
    def __init__(self, *args, **kwargs):
        self.logger = Logger.get_logger(self.__class__.__name__)
