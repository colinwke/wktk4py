"""
model helper
"""
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wktk import PdPrinter


def _is_fxx_format(d):
    import re
    for k in d:
        if re.match('f\d+', k) is None: return False
    return True


def show_feature_importance(feature_importance, feature_names=None, show_count=None, plot=False):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    if isinstance(feature_importance, dict):
        # xgboost
        if feature_names is not None:
            import re
            is_fxx_format = True
            for key in feature_importance:
                if re.match('f\d+', key) is None:
                    b_fxx_format = False
                    break

            if is_fxx_format:
                cols_name_map = dict((('f' + str(i)), val) for i, val in enumerate(feature_names))
                feature_importance = dict((cols_name_map[key], val) for key, val in feature_importance.items())

        feature_importance = pd.Series(feature_importance)
    else:
        # sk-learn feature important is a list
        feature_importance = pd.Series(feature_importance, index=feature_names)
    feature_importance.sort_values(ascending=False, inplace=True)
    # print count
    if show_count is not None:
        feature_importance_part = feature_importance[:show_count]
    else:
        feature_importance_part = feature_importance
    # print feature importance
    PdPrinter.print_full(feature_importance_part, 'feature_importance', None)
    # plot feature importance
    if plot is True:
        feature_importance_part.plot.bar()
        plt.show()
    return feature_importance


def get_percent_index(values, percents):
    values = np.sort(values)

    t = ()
    i, j = 0, 0
    while (i < len(values)) & (j < len(percents)):
        if values[i] < percents[j]:
            i += 1
        elif values[i] >= percents[j]:
            t += (i,)
            i += 1
            j += 1

    return t


def show_predict_information(values, reference_x=None, save_path=None):
    if reference_x is None:
        reference_x = len(values) // 2

    values = np.sort(values)
    # get predict information
    p25, p50, p75 = get_percent_index(values, [0.25, 0.5, 0.75])
    # print
    print('=== predict probability information ===')
    text1 = 'reference line: %d' % reference_x
    text4 = 'reference proba: %.4f' % values[reference_x]
    text2 = 'central deviation: %d' % (reference_x - p50)
    text3 = 'confidence interval: %d' % (p75 - p25)
    txt = ('\n'.join([text1, text4, text2, text3]))
    print(txt + '\n')

    # plot
    plt.figure()
    plt.plot(values)
    plt.title('_'.join([str(x) for x in [p25, p50, p75]]) + '  (' + str(reference_x) + ')')
    plt.xlabel('instance')
    plt.ylabel('probability')
    plt.plot((reference_x, reference_x), (values.max(), values.min()), 'k--')
    plt.plot((p25, p25), (values.max(), values.min()), 'r-.', alpha=0.5)
    plt.plot((p50, p50), (values.max(), values.min()), 'r--', alpha=0.75)
    plt.plot((p75, p75), (values.max(), values.min()), 'r-.', alpha=0.5)
    plt.figtext(0.6, 0.2, txt)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()


def turn_pn(train, frac=0.2):
    print('======== turn pn, frac: %f ========' % frac)
    count = train['label'].value_counts()
    print('1 : %.0f    # %d' % ((count[0] / count[1]), len(train)))
    train_ng = train[train['label'] == 0]
    train_ng = train_ng.sample(frac=(1 - frac))
    train = train.drop(train_ng.index, axis=0)
    count = train['label'].value_counts()
    print('1 : %.0f    # %d' % ((count[0] / count[1]), len(train)))

    return train


def get_values(train, test, d_cols):
    """
    d_cols = {
        'id': []
        'ft': []
        'lb': x
        ...
    }
    """
    # init feature columns
    if d_cols['ft'] is None:
        d_cols['ft'] = [x for x in train.d_cols if x not in (d_cols['id'] + [d_cols['lb']])]

    train_x = train[d_cols['ft']]
    train_y = train[d_cols['lb']]
    test_x = test[d_cols['ft']]

    print('======== data information ========')
    print('feature dimension   : %d' % train_x.shape[1])
    print('train instance count: %d' % train_x.shape[0])
    print('test  instance count: %d' % test_x.shape[0])
    pos_len = len(train_y[train_y == 1])
    neg_len = len(train_y) - pos_len
    print('[pos : neg] # [%d : %d] # [%.0f : %.0f]' % (pos_len, neg_len, 1, (neg_len / pos_len)))

    return train_x, train_y, test_x


def f_score(y_pred, y_true, threshold=0.5, on_label=1, keeper=[], bp=True):
    from sklearn.metrics import roc_auc_score

    # if y_pred.dtype is not np.int:
        # if bp: print('roc_auc_score: %.4f' % roc_auc_score(y_true, y_pred))
    y_pred = np.where(y_pred > threshold, 1, 0)
    y_pred_index, y_true_index = set(np.where(y_pred == on_label)[0]), set(np.where(y_true == on_label)[0])
    n_tp, n_pred, n_true = len(y_pred_index & y_true_index), len(y_pred_index), len(y_true_index)
    precise, recall = n_tp / n_pred, n_tp / n_true
    f1 = (2 * precise * recall) / (precise + recall)

    keeper.append([f1, precise, recall])
    if bp: print('[ F: %.4f, P: %.4f, R: %.4f ]  [ TP: %d, NP: %d, NT: %d ]'
                 % (f1, precise, recall, n_tp, n_pred, n_true))

    # confusion_matrix
    from sklearn.metrics import confusion_matrix
    # if bp: print('confusion_matrix[0, 1]:\n%s' % str(confusion_matrix(y_pred, y_true)))

    return f1
