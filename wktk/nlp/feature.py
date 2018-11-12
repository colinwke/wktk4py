"""
text classification feature
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def nb_norm(x_train, y_train, x_test):
    """ nb feature
        An Classifier is built over NB log-count ratios as feature values
            https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline-eda-0-052-lb
            paper: https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf
    """

    def pr(x, y_i, y):
        p = x[y == y_i].sum(0)
        return (p + 1) / ((y == y_i).sum() + 1)

    r = np.log(pr(x_train, 1, y_train) / pr(x_train, 0, y_train))
    r = sp.csr_matrix(r)
    x_train = x_train.multiply(r)
    x_test = x_test.multiply(r)

    return x_train, x_test


def get_text_feature(texts,
                     labels=None,
                     nrow_train=None,
                     vec='bow',
                     lowercase=False,
                     analyzer='word',
                     single_token=True,
                     ngram_range=(1, 1),
                     stop_words=None,
                     min_df=2,
                     binary=True,
                     select_k=None):
    """ text feature
        bow or tfidf
        and chi2 feature selection
    """
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2

    # keep single char as word
    if single_token:
        token_pattern = r"\b\w+\b"
    else:
        token_pattern = r"(?u)\b\w\w+\b"

    # choose vec
    if vec is 'bow':
        vec = CountVectorizer(
            lowercase=lowercase,
            analyzer=analyzer,
            ngram_range=ngram_range,
            stop_words=stop_words,
            min_df=min_df,
            token_pattern=token_pattern,
            binary=binary)
    elif vec is 'tfidf':
        vec = TfidfVectorizer(
            lowercase=lowercase,
            analyzer=analyzer,
            ngram_range=ngram_range,
            stop_words=stop_words,
            min_df=min_df,
            token_pattern=token_pattern,
            sublinear_tf=True)
    else:
        raise ValueError('vec must be bow or tfidf!')

    # get word vector
    feature = vec.fit_transform(texts)
    feature_names = vec.get_feature_names()

    # feature select
    if (labels is not None) and (select_k is not None):
        if nrow_train is not None:
            x_train = feature[:nrow_train, :]
            x_test = feature[nrow_train:, :]
            y_train = labels[:nrow_train]

            feature_selector = SelectKBest(chi2, k=select_k)
            x_train = feature_selector.fit_transform(x_train, y_train)
            feature_names = np.array(feature_names)[feature_selector.get_support()]

            x_test = feature_selector.transform(x_test)

            # combine train test
            import scipy.sparse as sp
            feature = sp.vstack([x_train, x_test])

        else:
            feature_selector = SelectKBest(chi2, k=select_k)
            feature = feature_selector.fit_transform(feature, labels)
            feature_names = np.array(feature_names)[feature_selector.get_support()]

    return feature, list(feature_names)

def predict_by_kfold(feature, labels, feature_names):
    results = []
    kf = KFold(n_splits=5, shuffle=True)
    proba_threshold = 0.5
    evaluate_label = 1
    lgb = True
    for train_index, test_index in kf.split(feature):
        x_train, x_test = feature[train_index], feature[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        if lgb:
            proba, fi = predict_by_lgb(x_train, y_train, x_test, feature_names, ret_fi=True)
            sentiment_table.append(fi)
        else:
            x_train, x_test = nb_norm(x_train, y_train, x_test)
            clf = LogisticRegression().fit(x_train, y_train)
            if hasattr(clf, 'predict_proba'):
                proba = clf.predict_proba(x_test)[:, 1]
            else:
                print('without predict proba, hard predict!')
                proba = clf.predict(x_test)

            # # feature importance
            if hasattr(clf, 'coef_'):
                sentiment_table.append(pd.Series(clf.coef_[0], index=feature_names).sort_values(ascending=False))
            elif hasattr(clf, 'feature_importances_'):
                sentiment_table.append(
                    pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False))

        # pred = np.where(proba > proba_threshold, 1, 0)
        f_score(proba, y_test, on_label=evaluate_label, threshold=proba_threshold)
        results.append(np.array([proba, y_test, proba, test_index]).T)
    results = np.concatenate(results)
    print('eval parms: eval_label: %d, threshold: %s' % (evaluate_label, proba_threshold))
    f_score(results[:, 0], results[:, 1], on_label=evaluate_label, threshold=proba_threshold)
    # show_predict_information(results[:, 2])

    return results


class SentimentTable(object):
    def __init__(self):
        self.tables = []

    def append(self, table):
        self.tables.append(table)

    def get_replace_dict(self, save=None):
        if len(self.tables) == 0: return {}
        data = pd.concat(self.tables, axis=1)

        # split words by sentiment
        words_pos = data.iloc[np.all(data.values > 0, axis=1)]
        words_neg = data.iloc[np.all(data.values < 0, axis=1)]
        words_umb = data[~data.index.isin(list(words_pos.index) + list(words_neg.index))]

        # calc impact mean
        words_pos = words_pos.mean(axis=1).sort_values(ascending=False).map(lambda x: '%+.2g' % x)
        words_neg = words_neg.mean(axis=1).sort_values(ascending=False).map(lambda x: '%+.2g' % x)
        words_umb = words_umb.mean(axis=1).sort_values(ascending=False).map(lambda x: 'u%+.2g' % x)

        # concat
        pieces = pd.concat([words_pos, words_umb, words_neg])
        if save is not None:
            pieces.to_csv(save, encoding='utf-8')

        return pieces.to_dict()
