# Create your tasks here
from __future__ import absolute_import, unicode_literals
import pandas as pd
import numpy as np
from django.http import HttpResponse
from celery_tasks.celery import app
from sklearn.externals import joblib
import lightgbm as lgbm
import json
import pickle
import datetime
from celery_tasks.fuhe.libs.get_data_fangrong import getData
from celery_tasks.fuhe.libs.create_features import GetFeatures
from celery_tasks.fuhe.libs.lgbm_model import LGBMODEL
from celery_tasks.fuhe.libs.other_models import *

@app.task(bind=True)
def origin_lgbmodel_train(self , number,preNum):
    dataset = getData(number)
    print("已取出第%d加企业数据"%number)
    print("正在创建特征......")
    ba = GetFeatures(dataset)
    new_features = ba.get_original_features()
    print("已完成特征工程")
    model = LGBMODEL(new_features, dataset)
    print("正在训练模型中.......")
    model.train_model()
    print("模型训练完毕！")

@app.task(bind=True)
def incre_predict(self , number,preNum):
    new_dataset = get_data(number)
    original_lgbm_model = joblib.load('lgbm_model.pkl')
    print("模型加载成功")
    params = {'boosting_type': 'gbdt',
              'objective': 'regression',
              'metric': 'mae',
              }
    ba = GetFeatures(new_dataset)
    features = ba.get_original_features()
    lgb_train = lgbm.Dataset(features, list(new_dataset[96 * 3:]))
    present_lgbm_model = lgbm.train(params, lgb_train, num_boost_round=1000, init_model=original_lgbm_model,
                                    verbose_eval=False, keep_training_booster=True)
    joblib.dump(present_lgbm_model, 'lgbm_model.pkl')
    print("模型微调完毕，并保存！")
    prediction = inner_former_prediction(preNum, new_dataset)
    print("预测完毕！")
    return prediction.to_json(orient='columns')


