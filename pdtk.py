import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def scale(df, std=False):
    """ 对df进行归一化 """
    scaler = StandardScaler() if std else MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)


def df2series(df, sep=''):
    """ 合并连接df的行, 转换为series """
    return pd.Series([sep.join(row.astype(str)) for row in df.values], index=df.index)
