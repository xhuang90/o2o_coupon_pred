# !/usr/bin/env python 
# -*- coding: utf-8 -*-
# @time: 2021-08-28 20:07
# @author: Hobey Wong
# @contact: hobey0712@gmail.com
# @file: data_preprocess.py
# @desc:

import pandas as pd
import numpy as np
from datetime import date
import time
from logs import logger
import warnings
warnings.filterwarnings('ignore')


# set global args
is_sample = False
original_data_dir = 'data/origin'
prep_data_dir = 'data/prep'


def read_data(file_name, rename_col=None, sample_sz=10000):
    """
    read local data file
    :param file_name: string, format is like xxx.csv
    :param rename_col: list
    :param sample_sz: int, size of sample data, default is 10k
    :return:
    """
    df = pd.read_csv('{0}/{1}'.format(original_data_dir, file_name), keep_default_na=True)
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


def save_data(df, file_name):
    """
    save processed data
    :param df:
    :param file_name:
    :return:
    """

    df.to_csv('{0}/{1}'.format(prep_data_dir, file_name), index=False)
    logger.info('{0}/{1} is saved with length {2}'.format(prep_data_dir, file_name, len(df)))


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
    # s = str(s)
    if np.isnan(s):
        return -1

    else:
        s = str(s)
        return int(s[4: 6])


def get_day(s):
    """
    get day of date
    :param s:s
    :return:
    """
    # s = str(s)
    if np.isnan(s):
        return -1

    else:
        s = str(s)
        return int(s[6: 8])


def get_diff_btw_dates(x, y):
    """
    get days difference btw tow dates
    :param x: int, date_received, format is like yyyyMMdd
    :param y: int, date_used, format is like yyyyMMdd
    :return:
    """

    if np.isnan(x) or np.isnan(y):
        return -1
    else:
        x = str(x)
        y = str(y)
        f_date = date(int(x[:4]), int(x[4:6]), int(x[6:8]))
        l_date = date(int(y[:4]), int(y[4:6]), int(y[6:8]))
        delta = l_date - f_date

        return delta.days


def get_label(x, y):
    """
    get label according to date_used and date_received
    :param x: int, date_received, format is like yyyyMMdd
    :param y: int, date_used, format is like yyyyMMdd
    :return:
    """
    def cal_dates_delta(a, b):
        """
        get days difference btw tow dates
        :param a:
        :param b:
        :return:
        """
        a = str(a)
        b = str(b)
        f_date = date(int(a[:4]), int(a[4:6]), int(a[6:8]))
        l_date = date(int(b[:4]), int(b[4:6]), int(b[6:8]))
        delta = (l_date - f_date).days

        return delta

    # case0. not used coupon or used date is out of range
    if np.isnan(x):
        return 0
    # case1. ??? received_date is out of range ???
    if np.isnan(y):
        return -1
    # case3. if used coupon within 15 days
    elif cal_dates_delta(x, y) <= 15:
        return 1  # positive
    # case4. not used within 15 ds
    else:
        return -1


def get_new_feats(df):
    """
    add new features
    :param df: DataFrame
    :return:
    """

    # full reduction related
    df['is_full_reduction'] = df['discount_rate'].apply(is_full_reduction)
    df['full_cond'] = df['discount_rate'].apply(get_full_reduction_cond)
    df['full_save'] = df['discount_rate'].apply(get_full_reduction_save)
    # discount rate formatted
    df['discount_rate'] = df['discount_rate'].apply(get_discount_rate)
    # missing distance
    df['distance'] = df['distance'].replace(np.nan, -1).astype(int)
    # date related
    df['month_received'] = df['date_received'].apply(get_month)
    df['month_used'] = df['date_used'].apply(get_month)
    # date_used and date_received
    df['days_gap'] = df.apply(lambda row: get_diff_btw_dates(row['date_received'], row['date_used']), axis=1)
    df['label'] = df.apply(lambda row: get_label(row['date_received'], row['date_used']), axis=1)

    return df


if __name__ == '__main__':

    logger.info('======== PREPROCESS START ========')
    t0 = time.time()
    # read original datafile
    # 1. offline training set
    df_off_train = read_data(file_name='ccf_offline_stage1_train.csv')
    # 2. online training set
    # df_on_train = read_data(file_name='ccf_online_stage1_train.csv')
    # 3. offline test set
    # rename_cols = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate',
    #                'distance', 'date_received']
    # df_off_test = read_data(file_name='ccf_offline_stage1_test_revised.csv', rename_col=rename_cols)

    # get features
    # 1. offline training set
    df_off_train_ = get_new_feats(df_off_train)
    save_data(df_off_train_, file_name='ccf_offline_stage1_train_prep.csv')
    # 2. online training set
    # df_on_train_ = get_new_feats(df_on_train)
    # save_data(df_on_train_, file_name='ccf_online_stage1_train_prep.csv')
    t1 = time.time()
    logger.info('process used time {0}s.'.format(round(t1 - t0), 3))
    logger.info('======== PREPROCESS END ========')
