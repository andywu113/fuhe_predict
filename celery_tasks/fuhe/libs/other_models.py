import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import lightgbm as lgbm
import pickle
import datetime
from sklearn import preprocessing
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.mixture import GaussianMixture
import warnings
from celery_tasks.fuhe.libs.get_data_fangrong import getData
from celery_tasks.fuhe.libs.create_features import GetFeatures
from celery_tasks.fuhe.libs.weatherServer import weather_info
from celery_tasks.fuhe.libs.xgb_model import XGBModel
from celery_tasks.fuhe.libs.lgbm_model import LGBMODEL
warnings.filterwarnings("ignore")
def get_data(number):
    """
    :param begin: 返回第几个企业的数据
    :return: dataset
    """
    dataset = getData(number)
    dataset = dataset
    return dataset

def get_features(dataset):
    """
    :param dataset:企业数据集
    :return: new_features
    """
    ba = GetFeatures(dataset)
    new_features = ba.get_original_features()
    return new_features
def next_node_predict(lgbm_model,dataset,number = 1):
    """
    :param number: the numbe we predict
    :param xgb_model: xgb_model which we have trained
    :return: (time,prediction)
    """
    tempTime = dataset.index[-1] + datetime.timedelta(minutes=15)
    dataset.loc[tempTime] = 0

    next_node_feature = []
    temp_node = dataset[-1]
    seasonal_feature = [dataset[-96 - 1], dataset[-96 * 2 - 1], dataset[-96 * 3 - 1]]
    t_range_feature = dataset[-96 * 1 - 1:-1]
    t_info_feature = [np.array(t_range_feature).mean(), np.array(t_range_feature).min(),
                          np.array(t_range_feature).max(), np.array(t_range_feature).std()]
    temp_index = dataset.index[-1]
    date_feature = [temp_index.day, temp_index.hour, temp_index.weekday()]
    next_node_feature.extend(seasonal_feature)
    next_node_feature.extend(t_range_feature)
    next_node_feature.extend(t_info_feature)
    next_node_feature.extend(date_feature)
    next_node_feature = [next_node_feature]
    next_node_prediction = lgbm_model.predict(next_node_feature)
    dataset.loc[tempTime] = next_node_prediction
    return (str(tempTime), next_node_prediction[0])

def inner_former_prediction(number,dataset):
    """
    :param number: the numbwe we want to predict
    :return: timeSeries
    """
    present_lgbm_model = joblib.load('lgbm_model.pkl')
    time = []
    prediction = []
    for i in range(number):
        one_node_prediction = next_node_predict(present_lgbm_model,dataset)
        time.append(pd.to_datetime(one_node_prediction[0]))
        prediction.append(one_node_prediction[1])
    return pd.Series(prediction, index=time)