# !/usr/bin/env python 
# -*- coding: utf-8 -*-
# @time: 2021-08-28 23:03
# @author: Hobey Wong
# @contact: hobey0712@gmail.com
# @file: __init__.py.py
# @desc:

"""
    日志模块标准范例
    文件结构为:
        logs:
            __init__.py
            ****.log
    日志包含两个handler:
        StreamHandler: 输出到控制台
        FileHandler: 输出到文件
"""

# packages
import os
import logging


# 初始化:日志等级、日志文件名
logger_level = logging.INFO
# 获取当前目录的绝对路径
# 注意:不能使用os.path.abspath('.') 这是运行的python文件的绝对路径
dir_path = os.path.dirname(os.path.abspath(__file__))
file_logger_name = os.path.join(dir_path, 'o2o_coupon_pred.log')

# 日志定义
logger = logging.getLogger(__name__)
logger.setLevel(logger_level)

# 创建 handler 输出到控制台
handler = logging.StreamHandler()
handler.setLevel(logger_level)

# 创建 handler 输出到文件
# a+: pro
# w+: dev
file_handler = logging.FileHandler(file_logger_name, mode='a+')
file_handler.setLevel(logger_level)

# 创建 logging format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)
logger.addHandler(file_handler)
