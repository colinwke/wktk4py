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


def get_content(line):
    """keep eng, chs, dig, and space.
    https://github.com/Shuang0420/Shuang0420.github.io/wiki/python-%E6%B8%85%E7%90%86%E6%95%B0%E6%8D%AE%EF%BC%8C%E4%BB%85%E4%BF%9D%E7%95%99%E5%AD%97%E6%AF%8D%E3%80%81%E6%95%B0%E5%AD%97%E3%80%81%E4%B8%AD%E6%96%87
    """
    line = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5]", ' ', line)

    return line


def is_only_chs(x):
    """判断x里面是否全为中文"""
    return re.search(r"[a-zA-Z0-9]", x) is None


if __name__ == '__main__':
    a = ["你是猪吗", "你是1头猪吗?", "你是a头猪吗?"]

    for i in a:
        print(is_only_chs(i))
