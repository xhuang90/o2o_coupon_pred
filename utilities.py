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


def add_agg_feat_names(df, df_grp, grp_cols, val_col, agg_ops, col_names):
    """

    :param df: DataFrame,
    :param df_grp: DataFrame,
    :param grp_cols: string,
    :param val_col: string, col to be stats
    :param agg_ops: string, process functions including count, mean, sum, std, min, max, nunique
    :param col_names: string,
    :return:
    """

    df_grp[val_col] = df_grp[val_col].astype('float')
    df_agg = pd.DataFrame(df_grp.groupby(grp_cols)[val_col].agg(agg_ops)).reset_index()

    df_agg.columns = grp_cols + col_names
    df = df.merge(df_agg, on=grp_cols, how='left')

    return df


def add_agg_feats(df, df_grp, grp_cols, val_col, agg_ops, kws):
    """

    :param df:
    :param df_grp:
    :param grp_cols:
    :param val_col:
    :param agg_ops:
    :param kws:
    :return:
    """

    col_names = []
    for op in agg_ops:

        col_names.append(kws + '_' + val_col + '_' + op)

    df = add_agg_feat_names(df, df_grp, grp_cols, val_col, agg_ops, col_names)

    return df


def add_count_new_feats(df, df_grp, grp_cols, new_feat_name):
    """
    process count features
    :param df: DataFrame,
    :param df_grp: DataFrame,
    :param grp_cols: string
    :param new_feat_name:
    :return:
    """

    df_grp[new_feat_name] = 1
    df_grp = df_grp.groupby(grp_cols).agg('sum').reset_index()
    # merge original df and new one
    df = df.merge(df_grp, on=grp_cols, how='left')

    return df


if __name__ == '__main__':

    pass
