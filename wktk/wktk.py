"""
wktk: wangke tool kits
2017-5-5
"""
import logging
import math
import multiprocessing as mp
import pickle
import sys
from os import makedirs
from os.path import exists, dirname
from time import time, ctime, strftime, localtime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Singleton(type):
    """https://stackoverflow.com/a/6798042/6494418"""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs)
        return cls._instances[cls]


# ==========================================================
# Timestamp
# ==========================================================

class Timestamp(object):
    def __init__(self):
        self._start = time()
        self._cstart = self._start

        print("Timestamp start: %s" % str(ctime()))

    def cut(self, info=None):
        current = time()
        run_time = time() - self._cstart
        self._cstart = current

        print("Timestamp cut: %s, %.2fs" % (ctime(), run_time))
        if info is not None: print(info)

    def end(self):
        run_time = time() - self._start
        print("Timestamp end: %s, %.2fs" % (ctime(), run_time))

    def exit(self, info=None):
        self.end()

        if info is not None: print(info)
        exit(1015)


# ==========================================================
# pandas print tool
# ==========================================================

class PdPrinter:
    @staticmethod
    def format_df(df, info=None, max_rows=10, max_colwidth=50):
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)
        content = '================  PdPrinter.print_full  ================\n'
        if info is not None:
            content = content + str(info) + '\n===\n'

        with pd.option_context(
                'display.max_rows', max_rows,
                'display.max_columns', None,
                'display.expand_frame_repr', False,
                'display.max_colwidth', max_colwidth):
            return content + str(df)

    @staticmethod
    def print_full(df, info=None, max_rows=10, max_colwidth=50):
        print(PdPrinter.format_df(df, info, max_rows, max_colwidth))

    @staticmethod
    def print_exit(df, info=None, max_rows=10):
        PdPrinter.print_full(df, info, max_rows)
        exit(1015)

    @staticmethod
    def detect_nan(df, b_exit=False):
        print('\n================  PdPrinter.detect_nan  ================')

        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)

        nan_columns = df.columns[pd.isnull(df).any()].tolist()
        com_columns = [x for x in df.columns if x not in nan_columns]
        print('all columns     : (count: %d, ratio: %.4f)  %s' % (
            len(df.columns), 1, str(list(df.columns))))
        print('nan columns     : (count: %d, ratio: %.4f)  %s' % (
            len(nan_columns), len(nan_columns) / len(df.columns),
            str(nan_columns)))
        print('complete columns: (count: %d, ratio: %.4f)  %s' % (
            len(com_columns), len(com_columns) / len(df.columns),
            str(com_columns)))
        # nan statistics
        print('=== row count %d' % len(df))
        print('%-20s %-10s %s' % ('column', 'nan count', 'nan ratio'))
        for column in nan_columns:
            n_nan = df[column].isnull().sum()
            print('%-20s %-10d %.4f' % (column, n_nan, n_nan / len(df)))
        if b_exit: exit(1015)


# ==========================================================
# pickle utils
# ==========================================================

class PickleUtils:
    @staticmethod
    def load_pickle(file_path):
        with open(file_path, 'rb') as f:
            ret = pickle.load(f)
            print('PickleUitls: load from %s success!' % file_path)
            return ret

    @staticmethod
    def dump_pickle(data, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            print('PickleUitls: save to %s success!' % file_path)

    @staticmethod
    def load_exist_pickle(file_path):
        if exists(file_path):
            return PickleUtils.load_pickle(file_path)
        else:
            return None

    @staticmethod
    def get_cache_file_path(file_path, append_folder='cache'):
        index_r = file_path.rindex('\\')
        folder_path = file_path[:(index_r + 1)] + append_folder + '\\'
        if not exists(folder_path):
            makedirs(folder_path)

        file_path = folder_path + file_path[(index_r + 1):][:-4] + '.pk'

        return file_path

    @staticmethod
    def read_cache_csv(file_path, update=False):
        pk_path = PickleUtils.get_cache_file_path(file_path, 'cache')

        if (not exists(pk_path)) | update:
            data = pd.read_csv(file_path)
            PickleUtils.dump_pickle(data, pk_path)
        else:
            data = PickleUtils.load_pickle(pk_path)

        return data


# ==========================================================
# MultiProcessing framework
# ==========================================================

class MultiProcess:
    """ examples:
        MultiProcess.map(func3, data, {'x': 1000})
        MultiProcess.apply(func3, data, (1000,))
        MultiProcess.map(len, data)
        MultiProcess.apply(len, data)

        apply_parallel(dfGrouped, func) reference:
        https://stackoverflow.com/questions/26187759/parallelize-apply-after-pandas-groupby
    """

    @staticmethod
    def apply(func, pipe, args=()):
        n_core = min(len(pipe), mp.cpu_count() - 1)
        with mp.Pool(n_core) as pool:
            ret_list = [pool.apply_async(func, (i,) + args).get() for i in pipe]
        return ret_list

    @staticmethod
    def map(func, pipe, args=None):
        if args is None:
            args = {}
        n_cpu = min(len(pipe), mp.cpu_count() - 1)
        with mp.Pool(n_cpu) as pool:
            if len(args) == 0:
                ret_list = pool.map(func, pipe)
            else:
                from functools import partial
                ret_list = pool.map(partial(func, **args), pipe)
        return ret_list

    @staticmethod
    def apply_parallel(dfGrouped, func):
        n_cpu = min(len(dfGrouped), mp.cpu_count() - 1)
        with mp.Pool(n_cpu) as p:
            ret_list = p.map(func, [group for name, group in dfGrouped])
        return pd.concat(ret_list)

    @staticmethod
    def apply_series_parallel(dfGrouped, func):
        n_cpu = min(len(dfGrouped), mp.cpu_count() - 1)
        with mp.Pool(n_cpu) as p:
            ret_list = p.map(func, [group for name, group in dfGrouped])
        return ret_list

    # other helper for mp
    @staticmethod
    def split_table(data, num=0):
        if num == 0:
            num = mp.cpu_count() - 1

        len_data = len(data)
        per_data = math.ceil(len_data / num)

        bucket = [x * per_data for x in range(num)]
        data = [data.iloc[x:x + per_data, :] for x in bucket]

        return data


# ==========================================================
# counter
# ==========================================================

class LengthCounter(object):
    def __init__(self, x, info=''):
        if isinstance(x, int):
            self.len_pre = x
        else:
            self.len_pre = len(x)
        print('=== LengthCounter %s ===' % info)
        print('LengthCounter: init (count: %d)' % self.len_pre)

    def count(self, x, info=None):
        if info is not None:
            print(': ' + info)

        if isinstance(x, int):
            len_tmp = x
        else:
            len_tmp = len(x)

        print('LengthCounter: (%d %+d) = %d' % (
            self.len_pre, len_tmp - self.len_pre, len_tmp))
        self.len_pre = len_tmp


# ==========================================================
# Logging to file
# ==========================================================


class Logger(metaclass=Singleton):
    def __init__(self):
        self.names = set()
        self.formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)-5.5s]  %(message)s",
            datefmt="[%m-%d %H:%M:%S]")

    def get_names(self):
        return self.names

    def _setup_logger(self, name="default", log_file=None):
        """refs:
        https://stackoverflow.com/a/46098711/6494418
        https://stackoverflow.com/a/11233293/6494418
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # steam handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(self.formatter)
        logger.addHandler(handler)

        # file handler
        if log_file is not None:
            makedirs(dirname(log_file), exist_ok=True)
            handler = logging.FileHandler(log_file)
            handler.setFormatter(self.formatter)
            logger.addHandler(handler)

        return logger

    def get_logger(self, name="default", log_file=None):
        if name in self.names:
            logger = logging.getLogger(name)
        else:
            self.names.add(name)
            logger = self._setup_logger(name, log_file)
            print("[Set logger] name: {0}, file: {1}".format(name, log_file))

        return logger


# ==========================================================

class UnsortTool:
    @staticmethod
    def var2str(v):
        """ variable list to string """
        # print('[\'' + v[1:-1].replace(', ', ',').replace(',', '\',\'') + '\']')
        if ',' not in v:
            v = '\',\''.join(v.split())
        else:
            v = '\',\''.join(''.join(v.split()).split(','))
        v = v.replace('[', '').replace(']', '')
        v = '[\'' + v + '\']'
        print(v)
        exit(1015)

    @staticmethod
    def get_current_time(simple=True):
        """get current time."""
        time_format = "%m%d%H%M%S" if simple else "%Y-%m-%d %H:%M:%S"
        return strftime(time_format, localtime())

    @staticmethod
    def drop_duplicates(values):
        seen = set()
        seen_add = seen.add

        return [x for x in values if not (x in seen or seen_add(x))]

    @staticmethod
    def get_index_mapped(values, return_map=False, keep_order=False):
        # drop duplicates and sort
        if keep_order:
            elements = UnsortTool.drop_duplicates(values)
        else:
            elements = set(values)

        # get index_map
        values_map = dict((v, k) for k, v in enumerate(elements))

        # replace valeus
        values_mapped = [values_map[i] for i in values]

        if return_map:
            return values_mapped, values_map
        else:
            return values_mapped

    @staticmethod
    def get_color_mapped(values):
        value_set = set(values)
        colors = plt.cm.get_cmap('Spectral')(np.linspace(0, 1, len(value_set)))
        colors = {k: v for k, v in zip(value_set, colors)}
        colors = [colors[i] for i in values]

        return colors


# ==========================================================


class PdUtils():
    @staticmethod
    def sample_with_value(df, column, value, frac, reindex=False):
        """
            对column列的某个值进行抽样
            如: column = sex, values = {man, woman}, 对man进行抽样, 抽样比率为0.2
            抽样的方法
            1. 对去除的数据进行抽样
            2. 去除抽样的数据
        """
        column_value_index = pd.Series(df[df[column] == value].index)
        if (0 < frac) & (frac < 1):
            delete_frac = 1 - frac
            delete_index = column_value_index.sample(frac=delete_frac)
        elif frac < len(column_value_index):
            delete_frac = len(column_value_index) - frac
            delete_index = column_value_index.sample(n=delete_frac)
        else:
            raise ValueError('sample frac error!')

        df = df[df.index.isin(delete_index) == False]
        if reindex: df.index = range(len(df))

        return df


# ==========================================================

class FileUtils:
    @staticmethod
    def create_if_not_exist_dir(path):
        """https://stackoverflow.com/a/12517490/6494418"""
        makedirs(dirname(path), exist_ok=True)


# ==========================================================
# update from 58 ctr/unsort/tool
# 2018-10-30
# ==========================================================


class Printer:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    LIGHT_PURPLE = '\033[94m'
    PURPLE = '\033[95m'
    END = '\033[0m'

    @classmethod
    def red(cls, s, **kwargs):
        print(cls.RED + s + cls.END, **kwargs)

    @classmethod
    def green(cls, s, **kwargs):
        print(cls.GREEN + s + cls.END, **kwargs)

    @classmethod
    def yellow(cls, s, **kwargs):
        print(cls.YELLOW + s + cls.END, **kwargs)

    @classmethod
    def lightPurple(cls, s, **kwargs):
        print(cls.LIGHT_PURPLE + s + cls.END, **kwargs)

    @classmethod
    def purple(cls, s, **kwargs):
        print(cls.PURPLE + s + cls.END, **kwargs)
        
# ==========================================================
