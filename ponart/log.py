import sys
import logging

def init_logging(level=None, stream=None):
    if level is None:
        level = logging.DEBUG
    if stream is None:
        stream = sys.stdout
    logging.basicConfig(
            format = '[%(asctime)s] (%(levelname)s) %(message)s',
            datefmt = '%Y/%m/%d %H:%M:%S',
            level = level,
            stream = stream,
            )

