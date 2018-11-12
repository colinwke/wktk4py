"""
多数据集的相对特征直方图
多数据集对比, 查看是否存在特征分布不同
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp  # 分布判断


class MutualDistributed(object):
    def __init__(self):
        self.data = {}

    def init_data(self, data):
        if isinstance(data, dict):
            self.data = data
            # to numpy array
            for key in self.data:
                self.data[key] = np.array(self.data[key])
        else:
            raise ValueError('dict needed!')

    def add_values(self, key, values):
        self.data[key] = np.array(values)

    def scale_height(self):
        """ height scale """
        max_count = max([len(x) for x in self.data.values()])
        for key in self.data:
            if len(self.data[key]) != max_count:
                self.data[key] = np.random.choice(self.data[key], size=max_count)

    def hist_one(self, bins=100, title=None, info=None, save_path=None):
        try:
            plt.figure()

            values_comb = np.concatenate(list(self.data.values()))
            heights, bins = np.histogram(values_comb, bins=bins)

            for key in self.data:
                plt.hist(self.data[key], bins, alpha=0.3, label=key)

            plt.legend(loc='best')

            title_info = 'distributed analysis' if title is None else title
            title_info += '\n%s' % str(info) if info is not None else ''
            plt.title(title_info)

            if save_path is not None:
                plt.savefig('%s/%s.png' % (save_path, title))
            else:
                plt.show()
        except Exception as e:
            print('plot failed! [%s]' % e)
        finally:
            plt.close()

    @staticmethod
    def _process_na_value(values):
        """ move nan value and calc nan ratio """
        values = np.array(values)
        values_len = len(values)
        values_dropna = values[~np.isnan(values)]
        values_dropna_len = len(values_dropna)
        return values_dropna, (values_len - values_dropna_len) / values_len

    def hist(self, dataset_dict, columns=None, scale_height=False, bins=100, save_path=None):
        """ batch plot for muti-feature """
        if columns is None:
            columns = list(dataset_dict[list(dataset_dict.keys())[0]].columns)
        if len(columns) == 1:
            save_path = None

        print('Columns to plot: %s' % columns)
        for idx, column in enumerate(columns):
            print('plot column: %d, %s' % (idx, column))
            # calculate for each columns
            try:
                """ try: not numeric column, nan column """
                dict_nars = {}
                for key in dataset_dict:
                    """ add data set """
                    values, values_nar = self._process_na_value(dataset_dict[key][column])
                    self.add_values(key, values)
                    dict_nars[key] = values_nar
                infos = []
                if max(dict_nars.values()) > 0:
                    infos.append(str({key: '%.4g' % dict_nars[key] for key in dict_nars}))

                if len(self.data) == 2:
                    """ calc ks_2samp """
                    x, y = self.data.values()
                    stat, pval = ks_2samp(x, y)
                    infos.append('{stat: %.4g, pval: %.4g}' % (stat, pval))

                if scale_height:
                    self.scale_height()

                info = None if len(infos) == 0 else ', '.join(infos)
                self.hist_one(bins=bins, title='%d_%s' % (idx, column), info=info, save_path=save_path)
            except Exception as e:
                print(e.__doc__)
                raise e


def main():
    # 配置路径
    folder = r'F:your_folder_path' + '\\'
    train = pd.read_csv(folder + 'verification_new1.csv')
    test = pd.read_csv(folder + 'test_new.csv')
    train_trade = train[train['is_trade'] == 1]
    train = train.drop('is_trade', axis=1)
    train_trade = train_trade.drop('is_trade', axis=1)

    mutual_d = MutualDistributed()
    dataset_dict = {
        'train': train._get_numeric_data(),
        'test': test._get_numeric_data(),
        'train_trade': train_trade._get_numeric_data()
    }
    columns = [list(test.columns)[138]]  # plot one column
    # columns = None  # plot all columns
    save_path = 'F:your_saveing_path'  # 图片保存的路径
    mutual_d.hist(dataset_dict, columns=columns, scale_height=True, save_path=save_path)


if __name__ == '__main__':
    main()
