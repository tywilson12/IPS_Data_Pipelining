from logging.handlers import TimedRotatingFileHandler
import logging
import sys
import os


def setLogger() -> logging.Logger:
    nameLogger = "HDAS_RAWDATA_2DMAP"
    logger = logging.getLogger(nameLogger)
    logger.setLevel(logging.DEBUG)
    folder = 'log'
    if not os.path.exists(folder):
        os.mkdir(folder)
    fh = TimedRotatingFileHandler(os.path.join(folder,
                                  '.'.join([nameLogger, folder])),
                                  when='midnight')
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(processName)s:%(threadName)s]'
                                  '%(message)s',  # .%(funcName)s
                                  datefmt='%d/%b/%Y %H:%M:%S')  # '%a, %d %b %Y %H:%M:%S'
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("####################  Started Programme ####################")
    return logger


LOGGER = setLogger()
