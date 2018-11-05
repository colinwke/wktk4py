from os import listdir
from os.path import isfile, join

from wktk import *
from bdc_util.pd_utils import df2list


def get_reference(path, file_suffix='.csv', return_filename=False):
    files = [f for f in listdir(path) if (file_suffix in f) & (isfile(join(path, f)))]
    files = files[::-1]

    pieces = [pd.read_csv(path + '\\' + file) for file in files]

    if return_filename:
        pieces = (pieces, files)

    return pieces


def get_inner_info(prediction, path_refs, file_suffix='.csv'):
    refs, name_refs = get_reference(path_refs, file_suffix=file_suffix, return_filename=True)

    prediction = df2list(prediction)
    refs = [df2list(x) for x in refs]

    set_predict = set(prediction)
    inner_count = [len(set_predict & set(ref)) for ref in refs]
    inner_ratio = [x / len(set_predict) for x in inner_count]

    ret = pd.DataFrame([inner_count, inner_ratio], columns=name_refs, index=['inner count', 'inner ratio'])
    PdPrinter.print_full(ret)

    return ret
