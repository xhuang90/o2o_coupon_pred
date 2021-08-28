# !/usr/bin/env python 
# -*- coding: utf-8 -*-
# @time: 2021-08-28 20:07
# @author: Hobey Wong
# @contact: hobey0712@gmail.com
# @file: data_preprocess.py
# @desc:

import pandas as pd


# set global args
is_sample = True


def read_data(file_name, rename_col=None, sample_sz=10000):
    """
    read local data file
    :param file_name: string, format is like xxx.csv
    :param rename_col: list
    :param sample_sz: int, size of sample data, default is 10k
    :return:
    """
    df = pd.read_csv('data/{0}'.format(file_name), keep_default_na=True)
    if not rename_col:
        df.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate',
                      'distance', 'date_received', 'date']
    else:
        df.columns = rename_col
    if not is_sample:
        pass
    else:  # construct sample data
        idx = list(df.sample(n=sample_sz, random_state=10).index)  # sample is for shuffling
        df = df.iloc[idx, :]

    print('length of df:', len(df))
    return df


def get_discount_rate(s):
    """
    format discount rate
    :param s:
    :return:
    """
    s = str(s)
    # case0. no discount
    if s == 'null':
        return -1
    s = s.split(':')
    # case1. this is discount rate like 0.95
    if len(s) == 1:
        return float(s[0])
    # case2. this is full reduction like 100:10(save 10 over 100)
    else:
        return 1.0 - float(s[1]) / float(s[0])


def is_full_reduction(s):
    """
    boolean for full reduction
    :param s:
    :return:
    """
    s = str(s)
    s = s.split(':')
    # case0. this is discount rate like 0.95
    if len(s) == 1:
        return 0
    # case1. this is full reduction like 100:10(save 10 over 100)
    else:
        return 1


def get_full_reduction_cond(s):
    """
    condition of full reduction
    :param s:
    :return:
    """
    s = str(s)
    s = s.split(':')
    # case0. this is discount rate like 0.95
    if len(s) == 1:
        return -1
    # case1. this is full reduction like 100:10(save 10 over 100)
    else:
        return int(s[0])


def get_full_reduction_save(s):
    """
    save of full reduction
    :param s:
    :return:
    """
    s = str(s)
    s = s.split(':')
    # case0. this is discount rate like 0.95
    if len(s) == 1:
        return -1
    # case1. this is full reduction like 100:10(save 10 over 100)
    else:
        return int(s[1])


def get_month(s):
    """
    get month of date
    :param s:
    :return:
    """

    if s[0] == 'null':
        return -1

    else:
        return int(s[4: 6])


def get_day(s):
    """
    get day of date
    :param s:
    :return:
    """

    if s[0] == 'null':
        return -1

    else:
        return int(s[6: 8])


def get_diff_btw_dates(s):

    pass


if __name__ == '__main__':

    pass
