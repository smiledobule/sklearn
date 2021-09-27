import numpy as np
import pandas as pd
from pandas import DataFrame


class MumBaby:
    def __init__(self):
        """
        user_id:用户身份信息

        auction_id:购买行为编号

        cat_id:商品种类序列号

        cat:商品序列号

        property:商品属性

        buy_mount:购买数量

        day:购买时间

        婴儿信息表格字段：

        birthday:出生日期

        gender:性别（0 male；1 female）
        """
        pass

    def mum_baby(self):
        info_df, record_df = self.__get_data()
        self.__buy_mount(record_df)
        self.__rate(record_df)
        pass

    @staticmethod
    def __rate(record_df):
        record_df['date'] = record_df.apply(lambda x: str(x['day'])[0:4], axis=1)
        group_df = record_df.groupby(['date'])['day'].count().reset_index(name='cnt')
        group_df['rate'] = group_df['cnt'].pct_change(periods=1)
        pass

    @staticmethod
    def __buy_mount(record_df):

        count = record_df.shape[0]
        mount_mean = record_df['buy_mount'].mean()
        mount_min = record_df['buy_mount'].min()
        mount_00 = record_df['buy_mount'].quantile(q=0.00)
        mount_25 = record_df['buy_mount'].quantile(q=0.25)
        mount_50 = record_df['buy_mount'].quantile(q=0.5)
        mount_75 = record_df['buy_mount'].quantile(q=0.75)
        mount_100 = record_df['buy_mount'].quantile(q=1)
        mount_max = record_df['buy_mount'].max()
        mount_std = record_df['buy_mount'].std()

        pass

    @staticmethod
    def __get_data():
        path_1 = '../data/mum_baby_trade/(sample)sam_tianchi_mum_baby.csv'
        path_2 = '../data/mum_baby_trade/(sample)sam_tianchi_mum_baby_trade_history.csv'

        info_df = pd.read_csv(path_1)
        record_df = pd.read_csv(path_2)
        # record_df = pd.merge(record_df, info_df, on=['user_id'], how='left')
        return info_df, record_df
        pass

    pass


if __name__ == '__main__':
    MumBaby().mum_baby()
    pass
