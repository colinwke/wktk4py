"""
feature merger
"""

import pandas as pd

from wktk import PdPrinter, PickleUtils


def merge_table(left, right):
    intersection_col = [val for val in left.columns if val in right.columns]
    print('intersection columns:', intersection_col)

    if len(intersection_col) > 2:  # error merge
        print('ERROR: intersection columns upper 2! preprocess end!')
        exit(2)

    mg = pd.merge(left, right, how='left', on=intersection_col)

    return mg


def merge_sequence_table(tables):
    left = None
    for table in tables:
        if left is None:
            left = table
        else:
            left = merge_table(left, table)

    return left


def merge_sequence_table_by_path(paths):
    left = None
    for path in paths:
        if left is None:
            left = PickleUtils.load_pickle(path)
        else:
            right = PickleUtils.load_pickle(path)
            left = merge_table(left, right)

    return left


def merge_and_save_sequence_table_by_path(paths, fillna=None, remove_chars=None, file_name_suffix='.pk'):
    # first path file is id columns
    left = merge_sequence_table_by_path(paths)

    if fillna is not None:
        left.fillna(fillna, inplace=True)

    # id columns
    ids_path = paths[0]
    feature_paths = paths[1:]
    feature_paths = [x.split('\\')[-1][:-len(file_name_suffix)] for x in feature_paths]

    if isinstance(remove_chars, list):
        paths_map = list(feature_paths)
        for r in remove_chars:
            feature_paths = [x.replace(r, '') for x in feature_paths]

    # print and save
    mg_table_path = ids_path[:-len(file_name_suffix)] + '_' + '_'.join(feature_paths) + file_name_suffix
    PdPrinter.print_full(left, 'save to: ' + mg_table_path, max_rows=5)
    PickleUtils.dump_pickle(left, mg_table_path)
