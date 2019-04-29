"""
wktk: wangke tool kits
2017-5-5
"""
import os
import sys
import time
import yaml
import math
import pickle
import logging
import itertools
import functools
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        self._start = time.time()
        self._cstart = self._start
        self.log = []

        msg = "Timestamp start: %s" % str(time.ctime())
        self.log.append(msg)
        print(msg)

    def cut(self, info=None):
        current = time.time()
        run_time = current - self._cstart
        self._cstart = current

        msg = "Timestamp cut: %s, %.2fs" % (time.ctime(), run_time)
        if info is not None: msg = "\n".join([msg, info])
        self.log.append(msg)
        print(msg)

    def end(self):
        run_time = time.time() - self._start
        msg = "Timestamp end: %s, %.2fs" % (time.ctime(), run_time)
        self.log.append(msg)
        print(msg)

    def exit(self, info=None):
        self.end()

        if info is not None: print(info)
        exit(1015)

    def get_log(self):
        return "\n".join(self.log)


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
            print('PickleUitls: load from `%s` success!' % file_path)
            return ret

    @staticmethod
    def dump_pickle(data, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            print('PickleUitls: save to `%s` success!' % file_path)

    @staticmethod
    def load_exist_pickle(file_path, remake=False):
        if os.path.exists(file_path) and not remake:
            return PickleUtils.load_pickle(file_path)
        else:
            return None


# ==========================================================
# MultiProcessing framework
# ==========================================================

class MultiProcess:
    """ refs:
    http://blog.adeel.io/2016/11/06/parallelize-pandas-map-or-apply/
    http://www.racketracer.com/2016/07/06/pandas-in-parallel/
    https://stackoverflow.com/a/716482/6494418
    https://stackoverflow.com/a/27027632/6494418
    """

    @staticmethod
    def _get_process_num_core(num_core=None):
        """get process core count.
        float value: make frac of available core, frac * max_core, range (0, 1);
        negative int value: available core to minus, range[0, 1-max_core];
        positive int value: core count;
        otherwise: available core minus 1.
        """
        cpu_count = multiprocessing.cpu_count()
        if num_core is None:
            num_core = multiprocessing.cpu_count() - 1
        elif isinstance(num_core, float):
            num_core = int(cpu_count * num_core)
        elif num_core <= 0:
            num_core = cpu_count + num_core

        if num_core > cpu_count or num_core < 1:
            raise ValueError(
                "MultiProcess core count error! available [1, %d], "
                "but get %s." % (cpu_count, num_core))

        return num_core

    @staticmethod
    def _map_pieces(func, pieces, *args, **kwargs):
        return [func(x, *args, **kwargs) for x in pieces]

    @staticmethod
    def map(func, data_list, num_core=None, single=False, tqdm=True, *args, **kwargs):
        if single:  # single core test
            print("[MultiProcess] single test!")
            if tqdm: data_list = Tqdm(data_list)
            return MultiProcess._map_pieces(func, data_list, *args, **kwargs)

        num_core = MultiProcess._get_process_num_core(num_core)

        print("[MultiProcess] use process core: %d" % num_core)
        data_list = np.array_split(data_list, num_core)

        # add tqdm for tail data block
        if tqdm: data_list = [x if i != num_core - 1 else Tqdm(x)
                              for i, x in enumerate(data_list)]

        with multiprocessing.Pool(num_core) as pool:
            data_list = list(itertools.chain.from_iterable(
                pool.map(functools.partial(
                    MultiProcess._map_pieces, func, *args, **kwargs), data_list)))

        return data_list


# ==========================================================
# counter
# ==========================================================

class LengthCounter(object):
    def __init__(self, x, tag="none"):
        if isinstance(x, int):
            self.len_pre = x
        else:
            self.len_pre = len(x)

        self.tag = tag
        print("[LengthCounter] (%s)(%d: init count)" % (tag, self.len_pre))

    def count(self, x, info=""):
        if isinstance(x, int):
            len_tmp = x
        else:
            len_tmp = len(x)

        print('[LengthCounter] (%s)(%d: %d%+d) %s' % (
            self.tag, len_tmp, self.len_pre, len_tmp - self.len_pre, info))
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
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
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
        return time.strftime(time_format, time.localtime())

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
        values_mapped = [values_map[i] for i in np.array(values)]

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

    @staticmethod
    def move_list_tail_to_head(l):
        if not isinstance(l, list):
            l = list(l)
        l.insert(0, l.pop(-1))
        return l

    @staticmethod
    def move_val2idx(lst, val, to_idx):
        if not isinstance(lst, list): lst = list(lst)
        if to_idx < 0:  to_idx = len(lst) + to_idx
        lst.pop(lst.index(val))
        lst.insert(to_idx, val)

        return lst

    @staticmethod
    def reverse_dict_map(d_map):
        return {v: k for k, v in d_map.items()}

    @staticmethod
    def get_var_size(var):
        var_size = sys.getsizeof(var)
        for unit in ["b", "k", "m", "g", "t"]:
            if var_size > 1024:
                var_size /= 1024
            else:
                print("[Variable Size] %f%s" % (var_size, unit))
                break

    @staticmethod
    def get_root_var_name(var):
        """https://stackoverflow.com/a/40536047/6494418"""
        for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
            if len(names) > 0:  return names[0]


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

    @staticmethod
    def rename_col(df, old_col, new_col):
        return df.rename(columns={old_col: new_col})

    @staticmethod
    def sort_df_by_str_len(df, str_col, ascending=True):
        """https://stackoverflow.com/a/46177383/6494418"""
        df['s8t1r0l0e7n'] = df[str_col].str.len()
        return df.sort_values(
            by=['s8t1r0l0e7n', str_col],
            ascending=ascending).drop('s8t1r0l0e7n', axis=1)

    @staticmethod
    def keep_duplicates(df, col):
        """https://stackoverflow.com/a/33381246/6494418"""
        return df[df[col].duplicated(keep=False)]


# ==========================================================

class FileUtils:
    @staticmethod
    def create_if_not_exist_dir(path):
        """https://stackoverflow.com/a/12517490/6494418"""
        os.makedirs(path.dirname(path), exist_ok=True)


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

class Yaml:
    """refs:
    https://stackoverflow.com/a/1774043/6494418
    https://stackoverflow.com/a/12471272/6494418
    """

    @staticmethod
    def load(file):
        with open(file, 'r') as stream:
            try:
                result = yaml.load(stream)
            except yaml.YAMLError as exc:
                raise yaml.YAMLError(exc)

        return result

    @staticmethod
    def dump(data, file):
        with open(file, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)


# ==========================================================
# update 2019-3-27 15:27:53
# ==========================================================

class Email:
    """python send email.
    ---
    refs:
    http://www.runoob.com/python3/python3-smtp.html
    https://stackoverflow.com/a/6270987/6494418
    https://blog.csdn.net/weixin_40475396/article/details/78693408
    https://blog.csdn.net/gpf951101/article/details/78909233
    https://blog.csdn.net/mouday/article/details/79896727
    """

    def __init__(self):
        self.mail_host = "smtp.163.com"
        self.mail_user = "wfcrgt@163.com"
        self.mail_pass = "wang1ke23ctrip45"

        self.smtpObj = None

    def _login(self):
        import smtplib

        self.smtpObj = smtplib.SMTP_SSL(self.mail_host, port=465)
        self.smtpObj.login(self.mail_user, self.mail_pass)

    def send_email(self, subject="none", content="none", receivers=None):
        from email.mime.text import MIMEText
        from email.header import Header

        if self.smtpObj is None:
            self._login()

        if receivers is None:
            receivers = self.mail_user

        message = MIMEText(content, "plain", "utf-8")
        message["Subject"] = Header(subject, "utf-8")
        message["From"] = self.mail_user  # same to sender
        message["To"] = str(receivers)

        print("\n".join(["=" * 32, str(message), "=" * 32]))
        self.smtpObj.sendmail(self.mail_user, receivers, message.as_string())
        print("send email success!")


class ArgsUtils:
    @staticmethod
    def str2bool(value):
        """string true, false to bool True, False."""
        value = value.lower()
        if value in ('true', 'false'):
            return v_lower == 'true'
        else:
            raise ValueError("string 2 bool error!")

    @staticmethod
    def get_args_info(args, argv):
        """get str args info.
        ---
        parser = argparse.ArgumentParser()
        parser.add_argument('--test_transform', type=str2bool, default='false')

        args, argv = parser.parse_known_args()
        args_info = get_args_info(args, argv)

        return args, args_info
        """
        args_info = "\n    ".join(
            ["==parsed args:"] + ['%s: %s' % x for x in vars(args).items()])
        argv_info = "\n    ".join(
            ["==unknown args:"] + (argv if argv else ["None"]))

        return "\n  ".join(["==[ARGS PARSER]", args_info, argv_info])


class Tqdm:
    def __init__(self, iterable, num_marker=10):
        self.iterable = iterable
        self.len = len(self.iterable)
        self.num_marker = num_marker
        self.num_gap = int(self.len / self.num_marker)
        # time
        self.start_time = time.time()
        self.cut_time = self.start_time
        # print format
        len_1, len_2 = len(str(self.num_marker)), len(str(self.len))
        self.format_print = "[tqdm] (%{}d/{} | %{}d/{}) %fs".format(
            len_1, self.num_marker, len_2, self.len)

    def cut(self):
        current = time.time()
        run_time = current - self.cut_time
        self.cut_time = current
        return run_time

    def __iter__(self):
        n = 0
        for obj in self.iterable:
            n += 1
            if n % self.num_gap == 0:
                print(self.format_print % (n // self.num_gap, n, self.cut()))
            yield obj
        print("[tqdm] end! runtime: %fs" % (time.time() - self.start_time))


# ===================================================================
# test functions
def add1(x):
    x = x + 1
    return x


if __name__ == '__main__':
    # l = range(1000000)
    # a1 = MultiProcess.map(add1, l)
    # a2 = MultiProcess.map(add1, l, single=True)
    #
    # print(pd.Series([x[0] == x[1] for x in zip(a1, a2)]).value_counts())
    UnsortTool.get_var_size(90)
