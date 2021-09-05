# !/usr/bin/env python 
# -*- coding: utf-8 -*-
# @time: 2021-09-05 14:29
# @author: Hobey Wong
# @contact: hobey0712@gmail.com
# @file: utilities.py
# @desc:

import pandas as pd
from logs import logger


def read_data(file_name, rename_col=None, sample_sz=10000, is_sample=False, data_dir=None):
    """
    read local data file
    :param file_name: string, format is like xxx.csv
    :param rename_col: list
    :param sample_sz: int, size of sample data, default is 10k
    :param is_sample: boolean,
    :param data_dir: string,
    :return:
    """
    if data_dir is None:

        raise ValueError(print('data_dir cannot be None.'))

    df = pd.read_csv('{0}/{1}'.format(data_dir, file_name), keep_default_na=True)
    if not rename_col:
        df.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate',
                      'distance', 'date_received', 'date_used']
    else:
        df.columns = rename_col
    if not is_sample:
        pass
    else:  # construct sample data
        idx = list(df.sample(n=sample_sz, random_state=10).index)  # sample is for shuffling
        df = df.iloc[idx, :]

    print('length of df:', len(df))

    return df


def save_data(df, file_name, data_dir):
    """
    save processed data
    :param df:
    :param file_name:
    :param data_dir:
    :return:
    """

    if data_dir is None:
        raise ValueError(print('data_dir cannot be None.'))

    df.to_csv('{0}/{1}'.format(data_dir, file_name), index=False)
    logger.info('{0}/{1} is saved with length {2}'.format(data_dir, file_name, len(df)))


if __name__ == '__main__':

    pass
