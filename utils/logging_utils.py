from datetime import datetime
import logging

logging_levels = {
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'ERROR': logging.ERROR,
    'WARN': logging.WARN,
}

def get_std_logging(filename, logging_level=logging.INFO):
    logging.basicConfig(
        filename=f'{filename}_{datetime.now()}.log',
        format='%(asctime)s %(filename)s: %(lineno)d [%(levelname)s] %(message)s',
        level=logging_level,
    )
    return logging
