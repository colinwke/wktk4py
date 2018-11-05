import pandas as pd
from bdc_util.model.model_helper import show_feature_importance

from wktk import PdPrinter


def predict_by_lr(x_train, y_train, x_test, ft_cols):
    from sklearn.linear_model import LogisticRegression

    # train model
    clf = LogisticRegression()
    clf = clf.fit(x_train, y_train)

    # feature importance
    feature_importance = pd.Series(clf.coef_[0], index=ft_cols).sort_values(ascending=False)
    PdPrinter.print_full(feature_importance, info='feature importance')

    # predict
    prediction = clf.predict_proba(x_test)

    return prediction[:, 1]


def predict_by_dt(x_train, y_train, x_test, cols_name):
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(max_depth=5)
    clf = clf.fit(x_train, y_train)
    predict = clf.predict_proba(x_test)
    show_feature_importance(clf.feature_importances_, cols_name, show_count=50)

    return predict[:, 1]


def predict_by_gbdt(x_train, y_train, x_test, cols_name):
    from sklearn.ensemble import GradientBoostingClassifier

    clf = GradientBoostingClassifier(learning_rate=0.01, n_estimators=300)
    clf.fit(x_train, y_train)
    show_feature_importance(clf.feature_importances_, cols_name, show_count=50)
    predict = clf.predict_proba(x_test)

    return predict[:, 1]


def predict_by_rf(x_train, y_train, x_test, cols_name):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=500, max_features='sqrt', max_depth=12)
    clf.fit(x_train, y_train)
    predict = clf.predict_proba(x_test)
    show_feature_importance(clf.feature_importances_, cols_name, show_count=50)

    return predict[:, 1]


def predict_by_xgb(x_train, y_train, x_test, feature_names=None):
    import xgboost as xgb

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test)

    params = {
        'silent': 1,
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.01,  # 0.1
        'max_depth': 3,  # 6
        'min_child_weight': 2,  # 2  # 4
        'colsample_bytree': 0.7,
        'subsample': 0.7,
        'scale_pos_weight': 1,
    }

    # train
    bst = xgb.train(params, dtrain, num_boost_round=1500)  # num_boost_round=1000
    # feature importance
    show_feature_importance(bst.get_score(), feature_names)
    # predict
    predict = bst.predict(dtest)

    return predict


def get_xgb_pred_leaf(x_train, y_train, x_test, n_tree=10):
    import xgboost as xgb

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test)

    params = {
        'silent': 1,
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.01,  # 0.1
        'max_depth': 3,  # 6
        'min_child_weight': 2,  # 2  # 4
        'colsample_bytree': 0.7,
        'subsample': 0.7,
        'scale_pos_weight': 0.15,
    }

    bst = xgb.train(params, dtrain, num_boost_round=n_tree)  # num_boost_round=1000

    ft_train = bst.predict(dtrain, pred_leaf=True)
    ft_test = bst.predict(dtest, pred_leaf=True)

    return ft_train, ft_test
