# @date: 2018/11/9 17:25
# @author: wangke
# @concat: wangkebb@163.com
# =========================
from wktk.wktk import LoggingConfig
import logging


def logger_example():
    LoggingConfig.init("../_test/logging_config.log")
    logger = logging.getLogger()

    logger.info("hello")


if __name__ == '__main__':
    logger_example()
