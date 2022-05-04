import logging
import sys
import os
import logging.handlers


"""
日志级别：
critical ： 50
error：40
warning：30
info：20
debug：10
notest ：0
"""


"""

Logger is also the first to filter the message based on a level — if you set the logger to INFO, 

and all handlers to DEBUG, 
you still won't receive DEBUG messages on handlers — they'll be rejected by the logger itself. 
If you set logger to DEBUG, but all handlers to INFO, 
you won't receive any DEBUG messages either — because while the logger says "ok, process this", 
the handlers reject it (DEBUG < INFO).
"""
#logger设定，输出到sys.stdout（也可设定输出到日志文件或者循环日志文件）



def get_logger(log_file=None, level=logging.DEBUG, when="D", backup=7,
             format="%(asctime)s.%(msecs)d %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s",
             datefmt="%m-%d %H:%M:%S"):
    if log_file:
        cur_dir = os.getcwd()
        formatter = logging.Formatter(format, datefmt)


        handler = logging.handlers.TimedRotatingFileHandler(os.path.join(cur_dir, log_file) + ".log",
                                                            when=when,
                                                            backupCount=backup)
        logger = logging.getLogger()
        logger.setLevel(level)

        handler.setLevel(level)
        handler.setFormatter(formatter)

        logger.addHandler(handler)
    else:
        logging.basicConfig(
            level=level,
            format=format,
            datefmt=datefmt
        )

        handler = logging.StreamHandler(sys.stderr)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    return logger

if __name__ == "__main__":

    log_file = "leah"
    logger = get_logger(log_file=log_file)
    logger.info("test1")
    logger.error("test2")
