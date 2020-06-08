import logging
import os
import socket
import sys
import time


def setup_logger(name, save_dir="", prefix="", timestamp=False):
    """
    Set up a logger
    :param name: The name of the logger
    :param save_dir: If provided, save the logger result to save_dir
    :param prefix: The prefix of log file
    :param timestamp: It true, saved log file name will include the timestamp
    :return:
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create console handler and set level to debug
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)

    # Create console handler and set level to debug, add formatter to ch
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        filename = 'log'
        if prefix:
            filename += '.' + prefix
        if timestamp:
            current_time = time.strftime('%m-%d_%H-%M-%S')
            filename += '.' + current_time + '.' + socket.gethostname()
        log_file = os.path.join(save_dir, filename + '.txt')

        # Create file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
