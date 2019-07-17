# 获取方融数据
# -*- coding: utf-8 -*-

from sqlalchemy import create_engine
import warnings
import time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# df['time_stamp'] = df['time_stamp'].astype(int)/1000
# df['time_stamp'].astype(int)
# pd.to_datetime(df.time_stamp,utc=True).tz_convert("Asia/Shanghai")


def convert_time(t):
    t = time.localtime(t)
    dt = time.strftime('%Y-%m-%d %H:%M:%S', t)
    return dt


def company_number(num, conn):
    sql = '''select * from `%s`''' % num
    df = pd.read_sql(sql, con=conn)
    df.time_stamp = df.time_stamp.values.astype('int64')/ 1000
    df.time_stamp = df.time_stamp.astype(int)
    df.index = df.time_stamp.apply(convert_time)
    df = df.drop(['time_stamp'], axis=1)

    # duiying=pd.read_excel('duiying.xlsx',sheet_name=0)
    # md=duiying[duiying.配电房名称==int(num)].mdmid.values[0]
    # print(md)

    # df=df[df.mdmid==md]['Meter.Ptotal']
    return df


def getData(i):
    duiying = pd.read_csv(r'C:\Users\42910\Desktop\fuhe_predict\lgbmmodel\libs\export.csv', encoding='utf-8')
    duiying = duiying.drop_duplicates()

    engine = create_engine("mysql://root:Ene!@#2019@192.168.1.157/fangrong_history?charset=utf8")
    conn = engine.connect()
    sql = '''show tables'''
    gongsi = pd.read_sql(sql, con=conn)
    # print(len(gongsi))
    num = gongsi.Tables_in_fangrong_history[i]
    # print(num)
    temp = duiying[duiying['name'] == int(num)]['mdmid'].values
    # print(temp)
    data = company_number(num, conn)
    mid = list(set(data.mdmid))
    data1 = {}
    for x in temp:

        temp1 = data[data.mdmid == x]['Meter.Ptotal'].drop_duplicates()
        temp1 = pd.to_numeric(temp1, errors='coerce')
        temp1.index = pd.to_datetime(temp1.index)
        temp1.sort_index(inplace=True)
        if len(temp1) != 0:
            data1[x] = temp1
        # data1=data1
    # data1=pd.DataFrame(data1)
    # data1=pd.to_numeric(data1, errors='coerce')
    # data1.index=pd.to_datetime(data1.index)
    # data1.sort_index(inplace=True)
    # return pd.DataFrame(data1)
    return pd.DataFrame(data1).sum(axis=1)