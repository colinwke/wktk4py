""" 使用正则过滤内容 """

import re


def filter_chs(s):
    """ 只保留中文字符, 使用空格分割 """
    s = re.sub('[^\u4e00-\u9fa5]', ' ', s)
    s = ' '.join(s.split())

    return s


def filter_eng(s):
    """ 只保留中文字符, 使用空格分割 """
    s = re.sub('[^a-zA-Z]', ' ', s).lower()
    s = ' '.join(s.split())

    return s


def filter_num(s):
    """ 只保留中文字符, 使用空格分割 """
    s = re.sub('[^0-9]', ' ', s)
    s = ' '.join(s.split())

    return s


def filter_out_html(s):
    """ 去除html标签 """
    s = re.sub('<.*?>', ' ', s)
    s = re.sub('&([0-9a-zA-Z]+);', '', s)
    s = ' '.join(s.split())

    return s


def get_all_numeric(s):
    """return a list extract all numeric, include int and float."""
    return [float(x) if '.' in x else int(x) for x in re.findall(r'-?\d+\.?\d*', s)]
