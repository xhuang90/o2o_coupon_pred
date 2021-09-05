# !/usr/bin/env python 
# -*- coding: utf-8 -*-
# @time: 2021-09-05 14:24
# @author: Hobey Wong
# @contact: hobey0712@gmail.com
# @file: feature_engineering.py
# @desc:


from utilities import read_data
import pandas as pd


def get_merchant_feats(df_feats):
    """
    extract merchant related features
    :param df_feats: DataFrame
    :return:
    """
    # extract merchant related data
    related_cols = ['merchant_id', 'coupon_id', 'distance', 'date_received', 'date_used']
    merchant = df_feats[related_cols].copy()

    ids = merchant[['merchant_id']].copy()
    # drop duplicated id
    ids.drop_duplicates(inplace=True)
    #
    # t1 = merchant


def read_df(data_dir, filename):
    """

    :param data_dir:
    :param filename:
    :return:
    """

    df_train = pd.read_csv('{0}/{1}'.format(data_dir, filename))
    # construct sample data
    idx = list(df_train.sample(n=1000, random_state=10).index)  # sample is for shuffling
    df = df_train.iloc[idx, :]

    return df


def main():

    df_train = read_df(prep_data_dir, filename='ccf_offline_stage1_train.csv')
    print(df_train.columns)
    get_merchant_feats(df_feats=df_train)


if __name__ == '__main__':

    is_sample = True
    prep_data_dir = 'data/prep'

    main()
