def df2list(df):
    tuple_columns = tuple()
    for column in df.columns:
        tuple_columns += (df[column],)

    return list(zip(*tuple_columns))