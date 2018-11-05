"""
submit result
"""
from bdc_util.submit.submit_helper import get_inner_info
from wktk import TimeUtils


def online_submit(data, folder_submit):
    print('probability description:')
    print(data['proba'].describe())

    file_time = TimeUtils.get_time()
    print('save as %s.csv' % file_time)

    # with probability
    data.to_csv(folder_submit + '_proba//' + file_time + '.csv', index=False)

    # submit file(without probability)
    data.drop('proba', axis=1).to_csv(folder_submit + file_time + '.csv', index=False)


def offline_submit(prediction, folder_reference):
    get_inner_info(prediction, folder_reference)


def submit(prediction, folder_submit, folder_reference, is_offline=True):
    if is_offline:
        offline_submit(prediction, folder_reference)
    else:
        online_submit(prediction, folder_submit)
