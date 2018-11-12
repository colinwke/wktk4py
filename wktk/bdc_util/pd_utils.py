def df2list(df):
    """return a list of tuple."""
    tuple_columns = tuple()
    for column in df.columns:
        tuple_columns += (df[column],)

    return list(zip(*tuple_columns))