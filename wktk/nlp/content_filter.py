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


def filter_parentheses_content(x):
    """remove content of (*)."""
    idx_l, idx_r = x.find('('), x.rfind(')')
    if idx_l != -1 and idx_r != -1:
        x = x[:idx_l] + x[idx_r + 1:]
    return x


def get_parentheses_content(x):
    """get content of (*)."""
    idx_l, idx_r = x.find('('), x.rfind(')')
    if idx_l != -1 and idx_r != -1:
        x = x[idx_l + 1:idx_r]
    return x


if __name__ == '__main__':
    l = """蓬莱一品民俗家庭公寓
蓬莱金萍渔家乐
蓬莱凯莱宾馆
速8酒店(蓬莱蓬莱阁登州路店)
蓬莱仙境姐姐家家庭驿站
蓬莱秋实家庭公寓
烟台阳光味道特色民宿
烟台琦秀度假公寓(沿河街分店)
烟台无限风光在木石99公寓(观音苑分店)
烟台仙境爱侣行汐•遇公寓(海洋极地世界分店)
"""
    l = l.split()
    print(l)

    for i in l:
        print(filter_parentheses_content(i))
