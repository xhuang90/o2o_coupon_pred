# !/usr/bin/env python 
# -*- coding: utf-8 -*-
# @time: 2021-09-05 14:24
# @author: Hobey Wong
# @contact: hobey0712@gmail.com
# @file: feature_engineering.py
# @desc:

import utilities as utils
import pandas as pd


def get_merchant_feats(df_feats):
    """
    extract merchant related features
    separate feature DataFrame and original DataFrame
    :param df_feats: DataFrame
    :return: DataFrame, with features of merchant
    """
    # extract merchant related data
    related_cols = ['merchant_id', 'coupon_id', 'distance', 'date_received', 'date']
    df_merchant = df_feats[related_cols].copy()
    # get merchant ids then drop duplicates
    ids = df_merchant[['merchant_id']].copy()
    ids.drop_duplicates(inplace=True)

    # feat1. count of transaction for each merchant
    f1 = df_merchant[~df_merchant.date.isna()][['merchant_id']].copy()
    df_merchant_feats = utils.add_count_new_feats(df=ids, df_grp=f1, grp_cols='merchant_id',
                                                  new_feat_name='total_sales')

    # feat2. count of transaction with coupon for each merchant
    f2 = df_merchant[(~df_merchant.date.isna()) & (~df_merchant.coupon_id.isna())][['merchant_id']].copy()
    df_merchant_feats = utils.add_count_new_feats(df=df_merchant_feats, df_grp=f2, grp_cols='merchant_id',
                                                  new_feat_name='sales_with_coupon')

    # feat3. count of distributed coupon for each merchant
    f3 = df_merchant[~df_merchant.coupon_id.isna()][['merchant_id']].copy()
    df_merchant_feats = utils.add_count_new_feats(df=df_merchant_feats, df_grp=f3, grp_cols='merchant_id',
                                                  new_feat_name='total_coupon')

    # feat4. max, min, mean, median of user distance for each merchant with used coupon
    f4 = df_merchant[(~df_merchant.date.isna()) & (~df_merchant.coupon_id.isna())
                     & (~df_merchant.distance.isna())][['merchant_id', 'distance']].copy()
    f4.distance = f4.distance.astype('int')
    agg_opts = ['max', 'min', 'mean', 'median']
    df_merchant_feats = utils.add_agg_feats(df=df_merchant_feats, df_grp=f4, grp_cols=['merchant_id'],
                                            val_col='distance', agg_ops=agg_opts, kws='merchant')

    # feat5. how much percentage of coupon being used for each merchant
    df_merchant_feats['sales_with_coupon'].fillna(0, inplace=True)
    df_merchant_feats['coupon_used_rate'] = \
        df_merchant_feats.sales_with_coupon.astype('float')/df_merchant_feats.total_coupon

    # feat6. how much percentage of transaction using coupon for each merchant
    df_merchant_feats['trans_with_coupon_rate'] = \
        df_merchant_feats.sales_with_coupon.astype('float') / df_merchant_feats.total_sales

    df_merchant_feats['total_coupon'].fillna(0, inplace=True)

    print(df_merchant_feats.head(5))

    return df_merchant_feats


def get_user_feats(df_feats):
    """
    extract user related features
    separate feature DataFrame and original DataFrame
    :param df_feats: DataFrame, all data to extract features
    :return: DataFrame, with features of users
    """

    # extract user related data
    related_cols = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate',
                    'distance', 'date_received', 'date']
    df_user = df_feats[related_cols].copy()
    # get merchant ids then drop duplicates
    ids = df_user[['user_id']].copy()
    ids.drop_duplicates(inplace=True)

    # feat1. count of transacted merchant for each user
    f1 = df_user[~(df_user.date.isna())][['user_id', 'merchant_id']].copy()
    f1.drop_duplicates(inplace=True)  # keep unique user-merchant pair
    df_user_feats = utils.add_count_new_feats(df=ids, df_grp=f1[['user_id']], grp_cols='user_id',
                                              new_feat_name='count_merchant')

    # feat2. max, min, mean, median of user distance for each user using coupon
    f2 = df_user[(~df_user.date.isna()) & (~df_user.coupon_id.isna())
                 & (~df_user.distance.isna())][['user_id', 'distance']].copy()
    f2.distance = f2.distance.astype('int')
    agg_opts = ['max', 'min', 'mean', 'median']
    df_user_feats = utils.add_agg_feats(df=df_user_feats, df_grp=f2, grp_cols=['user_id'],
                                        val_col='distance', agg_ops=agg_opts, kws='user')

    # feat3. count of transaction with coupon for each user
    f3 = df_user[(~df_user.date.isna()) & (~df_user.coupon_id.isna())][['user_id']].copy()
    df_user_feats = utils.add_count_new_feats(df=df_user_feats, df_grp=f3, grp_cols='user_id',
                                              new_feat_name='pay_with_coupon')

    # feat4. count of transaction of each user
    f4 = df_user[~df_user.date.isna()][['user_id']].copy()
    df_user_feats = utils.add_count_new_feats(df=df_user_feats, df_grp=f4, grp_cols='user_id',
                                              new_feat_name='pay_total')

    # feat5. count of receiving coupon of each user
    f5 = df_user[~df_user.coupon_id.isna()][['user_id']].copy()
    df_user_feats = utils.add_count_new_feats(df=df_user_feats, df_grp=f5, grp_cols='user_id',
                                              new_feat_name='coupon_received')

    # feat6. max, min, mean, median of day gap between receiving and using coupon
    f6 = df_user[(~df_user.date.isna()) & (~df_user.date_received.isna())
                 & (~df_user.coupon_id.isna())][['user_id', 'date', 'date_received']].copy()
    f6['day_gap'] = f6.apply(lambda row: utils.get_diff_btw_dates(row.date_received, row.date), axis=1)
    df_user_feats = utils.add_agg_feats(df=df_user_feats, df_grp=f6, grp_cols=['user_id'],
                                        val_col='day_gap', agg_ops=agg_opts, kws='user')

    df_user_feats['count_merchant'].fillna(0, inplace=True)

    # feat7. how much percentage of transaction using coupon for each user
    df_user_feats['pay_with_coupon'].fillna(0, inplace=True)
    df_user_feats['user_trans_with_coupon_rate'] = \
        df_user_feats.pay_with_coupon.astype('float') / df_user_feats.pay_total

    # feat7. how much percentage of coupon being used for each user
    df_user_feats['user_coupon_used_rate'] = \
        df_user_feats.pay_with_coupon.astype('float') / df_user_feats.coupon_received.astype('float')

    df_user_feats['pay_total'].fillna(0, inplace=True)
    df_user_feats['coupon_received'].fillna(0, inplace=True)

    return df_user_feats


def get_user_merchant_feats(df_feats):
    """
    extract features between user and merchant
    separate feature DataFrame and original DataFrame
    :param df_feats: DataFrame, all data to extract features
    :return: DataFrame, with features of users
    """
    # get user-merchant ids then drop duplicates
    ids = df_feats[['user_id', 'merchant_id']].copy()
    ids.drop_duplicates(inplace=True)

    # feat1. count of transaction between each user and merchant
    f1 = df_feats[['user_id', 'merchant_id', 'date']].copy()
    f1 = f1[~f1['date'].isna()][['user_id', 'merchant_id']]
    df_user_merchant = utils.add_count_new_feats(df=ids, df_grp=f1, grp_cols=['user_id', 'merchant_id'],
                                                 new_feat_name='user_merchant_pay_count')

    # feat2. count of receiving coupon
    print(df_user_merchant.head())


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
    get_user_merchant_feats(df_feats=df_train)


if __name__ == '__main__':

    is_sample = True
    prep_data_dir = 'data/prep'

    main()
