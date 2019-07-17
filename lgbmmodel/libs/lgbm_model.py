import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre_processing
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
from sklearn.externals import joblib
import pickle
import datetime
import seaborn as sns
import xgboost as xgb
import lightgbm as lgbm


class LGBMODEL(object):
    def __init__(self, new_features, dataset):
        """
        :param new_features: we use new_features as x_train
        :param dataset: we use dataset[96*3:] as y_train
        """
        self.new_features = new_features
        self.dataset = dataset

    def features_score(self):
        """
        :return: features name and their score
        """
        feature_names = ["seasona_3", "seasonal_2", "seasonal_1"]
        feature_names.extend(["t_%d" % i for i in range(1, 97)])
        feature_names.extend(["t_min", "t_max", "t_mean", "t_std"])
        feature_names.extend(["day", "hour", "weekday"])
        train_x = self.new_features
        train_y = self.dataset[96 * 3:]

        randomfrost = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, n_estimators=10)
        randomfrost.fit(train_x, train_y)
        print("Features sorted by their score:")
        importance = sorted(zip(map(lambda x: round(x, 4), randomfrost.feature_importances_), feature_names),
                            reverse=True)
        print(importance)
        names = [i[1] for i in importance]
        scores = [i[0] for i in importance]

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        width = 0.5
        idx = np.arange(len(names))
        plt.figure(figsize=(30, 7), dpi=80)
        plt.bar(idx, scores, width)
        plt.xticks(idx + width / 2, names, rotation=40)
        plt.title("特征得分情况")
        # plt.savefig("feature_score.png")
        plt.legend()
        plt.show()

    def rf_cv(self, max_depth, num_leaves, learning_rate, n_estimators, min_child_samples, subsample, reg_alpha,
              reg_lambda):
        """
        :param max_depth:每棵树最大深度
        :param num_leaves: 叶子节点个数
        :param learning_rate: 学习率
        :param n_estimators: 树的个数
        :param subsample: 子采样比例
        :param min_child_samples:最小节点样本数
        :param reg_alpha:叶子节点数惩罚系数
        :param reg_lambda:模型复杂度惩罚系数
        :return:-RMSE
        """
        train_x = self.new_features
        train_y = list(self.dataset[96 * 3:])
        lgb_train = lgbm.Dataset(train_x, train_y)
        params = {'max_depth': int(max_depth), 'num_leaves': int(num_leaves), 'learning_rate': learning_rate,
                  'n_estimators': int(n_estimators), 'min_child_samples': int(min_child_samples),
                  'subsample': subsample, 'reg_alpha': reg_alpha,
                  'reg_lambda': reg_lambda, 'random_state': 2, 'seed': 0
                  }

        lgbm_model = lgbm.train(params, lgb_train)
        predict_y = lgbm_model.predict(self.new_features)
        RMSE = np.sqrt(mean_squared_error(self.dataset[96 * 3:], predict_y))
        return -RMSE

    def optimize_parameters(self):
        rf_bo = BayesianOptimization(
            self.rf_cv,
            {'max_depth': (1, 20),
             'num_leaves': (5, 50),
             'learning_rate': (0.001, 1),
             'n_estimators': (100, 1000),
             'subsample': (0.001, 1),
             'min_child_samples': (50, 200),
             'reg_alpha': (0, 1),
             'reg_lambda': (0, 1)}
        )

        rf_bo.maximize()
        target = []
        for i in range(len(rf_bo.res)):
            target.append(rf_bo.res[i]['target'])
        params = [i for i in range(len(target)) if target[i] == max(target)][0]
        best_params = rf_bo.res[params]['params']
        #         best_params = rf_bo.max["params"]
        print(best_params)

    def train_model(self):
        """
        :return: XGBModel
        """
        train_x = self.new_features
        train_y = list(self.dataset[96 * 3:])
        lgb_train = lgbm.Dataset(train_x, train_y)
        params = {
            'objective': 'regression',
            'max_depth': 20,
            'num_leaves': 50,
            'learning_rate': 0.263,
            'n_estimators': 704,
            'min_child_samples': 52,
            'subsample': 0.1313,
            'colsample_bytree': 1,
            'reg_alpha': 0.3313,
            'reg_lambda': 0.9226
        }

        lgbm_model = lgbm.train(params, lgb_train)
        predict_y = lgbm_model.predict(train_x)
        Rmse = np.sqrt(mean_squared_error(train_y, predict_y))
        # print("RMSE为：%4f" % (Rmse))

        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['axes.unicode_minus'] = False
        # plt.figure(figsize=(20, 7), dpi=80)
        # plt.plot(train_y, label="原始序列")
        # plt.plot(predict_y, label="预测序列")
        # plt.title("模型训练情况")
        # plt.legen
        # plt.show()
        joblib.dump(lgbm_model, 'lgbm_model.pkl')

    #         return lgbm_model

    def next_Node_predict(self, lgbm_model, number=1):
        """
        :param number: the numbe we predict
        :param xgb_model: xgb_model which we have trained
        :return: (time,prediction)
        """
        tempTime = self.dataset.index[-1] + datetime.timedelta(minutes=15)
        self.dataset.loc[tempTime] = 0

        next_node_feature = []
        temp_node = self.dataset[-1]
        seasonal_feature = [self.dataset[-96 - 1], self.dataset[-96 * 2 - 1], self.dataset[-96 * 3 - 1]]
        t_range_feature = self.dataset[-96 * 1 - 1:-1]
        t_info_feature = [np.array(t_range_feature).mean(), np.array(t_range_feature).min(),
                          np.array(t_range_feature).max(), np.array(t_range_feature).std()]
        temp_index = self.dataset.index[-1]
        date_feature = [temp_index.day, temp_index.hour, temp_index.weekday()]
        next_node_feature.extend(seasonal_feature)
        next_node_feature.extend(t_range_feature)
        next_node_feature.extend(t_info_feature)
        next_node_feature.extend(date_feature)
        next_node_feature = [next_node_feature]
        next_node_prediction = lgbm_model.predict(next_node_feature)
        self.dataset.loc[tempTime] = next_node_prediction
        return (str(tempTime), next_node_prediction[0])

    def former_prediction(self, number):
        """
        :param number: the numbwe we want to predict
        :return: timeSeries
        """
        present_lgbm_model = joblib.load('lgbm_model.pkl')
        time = []
        prediction = []
        for i in range(number):
            one_node_prediction = self.next_Node_predict(present_lgbm_model)
            time.append(pd.to_datetime(one_node_prediction[0]))
            prediction.append(one_node_prediction[1])
        return pd.Series(prediction, index=time)

    def incremental_train_predict(self, GetFeatures, new_dataset,number=96):
        """
        :param GetFeatures: package which is used to create features
        :param new_dataset: new dataset to train lgbm_model
        :return: new lgbm_model
        """
        original_lgbm_model = joblib.load('lgbm_model.pkl')
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
        prediction = self.former_prediction(number)
        return prediction