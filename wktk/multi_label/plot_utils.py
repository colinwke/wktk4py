"""
样本的可视化
通过样本的特征出描述样本

1. 单个样本的可视化
2. 多个样本的可视化

---
划分了数据过后, 对所有数据进行聚类处理, 查看聚类处理后的结果
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.utils import check_array
from sklearn.preprocessing import scale
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

from wktk.wktk import PdPrinter


def encode_label(target):
    """ trans to bin series and to decimal """
    return pd.Series([''.join(row.astype(str)) for row in target.values], index=target.index)


def plot_target_corr_info(target):
    """ target 标签的相关性分析
    https://stackoverflow.com/questions/3529666/matplotlib-matshow-labels
    https://stackoverflow.com/questions/29432629/correlation-matrix-using-pandas
    """
    corr = target.corr()
    corr_ = corr.applymap(lambda x: 0 if x == 1 else x).abs()
    PdPrinter.print_full(corr_.max(axis=0).sort_values(), info='target max corr', max_rows=None)
    PdPrinter.print_full(target.corr().applymap(lambda x: '%.4g' % x), max_rows=None)
    plt.matshow(corr)
    plt.xticks(range(target.shape[1]), list(corr.columns), rotation='vertical')
    plt.yticks(range(target.shape[1]), list(corr.columns))
    plt.colorbar()
    plt.show()
    exit()


def plot_scatter_line(feature, label, light_label=1):
    feature = check_array(feature)
    label = np.array(label)

    feature = scale(feature)

    label_set = set(label)
    if len(label_set) == 2:
        colors = ['green' if x == 1 else 'gray' for x in label]
    else:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(label_set)))
        colors_map = dict((k, v) for v, k in enumerate(label_set))
        colors = [colors[colors_map[k]] for k in label]

    alphas = [0.7 if x == light_label else 0.3 for x in label]
    feature_indexes = list(range(feature.shape[1]))
    for i, label_temp in enumerate(label):
        if label_temp == light_label: continue
        plt.plot(feature_indexes, feature[i, :], c=colors[i], alpha=alphas[i])

    for i, label_temp in enumerate(label):
        if label_temp != light_label: continue
        plt.plot(feature_indexes, feature[i, :], c=colors[i], alpha=alphas[i])

    plt.show()


def plot_scatter(feature, label=None, title=None, show_index=None, dr_method='pca', n_components=2, save_path=None):
    feature = check_array(feature)

    # dimensionality reduction
    if feature.shape[1] > n_components:
        if dr_method == 'mds':
            dr_method = MDS(n_components=n_components, max_iter=99, n_init=1)
        else:
            dr_method = PCA(n_components=n_components)
        feature = dr_method.fit_transform(scale(feature))

    # label to color
    if label is None:
        label = 'k'
        label_count = 1
    else:
        label_set = set(label)
        label_count = len(label_set)
        colors = plt.cm.get_cmap('rainbow')(np.linspace(0, 1, len(label_set)))
        colors = {k: v for k, v in zip(label_set, colors)}
        label = [colors[i] for i in label]

    # plot instance
    fig = plt.figure()
    if n_components != 2:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(feature[:, 0], feature[:, 1], feature[:, 2], c=label)
    else:
        ax = fig.add_subplot(111)
        ax.scatter(x=feature[:, 0], y=feature[:, 1], c=label)

        # plot index
        if show_index is not None:
            if show_index is True:
                show_index = list(range(feature.shape[0]))

            for index, x, y in zip(show_index, feature[:, 0], feature[:, 1]):
                plt.annotate(index, (x, y), alpha=0.15)

    title_text = 'instance count: %d, label count: %d' % (feature.shape[0], label_count)
    if title is not None:
        title_text = '\n'.join([title, title_text])
    plt.title(title_text)

    if save_path is not None:
        plt.savefig('%s\%s.svg' % (save_path, title), format='svg')

    # plt.show()


def makedirs_if_not_exists(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)


def plot_instance(instance, info='plot instance', save_path=None):
    if ~isinstance(instance, np.ndarray): instance = np.array(instance)

    plt.figure(figsize=(14, 7))
    plt.title(info)
    plt.plot(instance, marker='o')
    plt.grid(True)

    if save_path is not None:
        makedirs_if_not_exists(save_path)
        plt.savefig('%s/%s.png' % (save_path, info))
        plt.close()
    else:
        plt.show()


def plot_multiple(instances, ids=None, info='plot_multiple', labels=None, save_path=None):
    instances = check_array(instances)

    if ids is None:
        ids = list(range(instances.shape[0]))
    else:
        instances = instances[ids, :]
        PdPrinter.print_full(instances)

    if labels is None:
        labels = ids
    else:
        labels = ['%04d ' % x for x in ids] + labels

    # plot
    plt.figure(figsize=(14, 7))
    for row, label in zip(instances, labels):
        plt.plot(row, marker='o', label=label, alpha=0.5)

    plt.title(info)
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        makedirs_if_not_exists(save_path)
        plt.savefig('%s/%s.png' % (save_path, info))
        plt.close()
    else:
        plt.show()


def plot_batch_one(ids, data, target, save_path=None):
    # load dataset

    data = scale(data)[ids, :]
    target = encode_label(target).map(lambda x: '%s %d' % (x, int(x, 2))).values[ids]
    target = ['%04d ' % x for x in ids] + target

    for i in range(len(data)):
        plot_instance(data[i, :], target[i], save_path)


if __name__ == '__main__':
    # data
    data = pd.DataFrame(np.random.rand(100, 10))
    # labels
    target = pd.DataFrame(np.where(np.random.rand(100, 10) > 0.5, 1, 0))

    # plot_batch_one([1, 2, 3], data, target)
    plot_instance(data.iloc[5, :])

