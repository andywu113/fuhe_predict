import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre_processing
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
import datetime
import seaborn as sns
import xgboost as xgb


class XGBModel(object):
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

        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['axes.unicode_minus'] = False
        # width = 0.5
        # idx = np.arange(len(names))
        # plt.figure(figsize=(30, 7), dpi=80)
        # plt.bar(idx, scores, width)
        # plt.xticks(idx + width / 2, names, rotation=40)
        # plt.title("特征得分情况")
        # # plt.savefig("feature_score.png")
        # plt.legend()
        # plt.show()
        return importance

    def rf_cv(self, learning_rate, n_estimators, max_depth, min_child_weight, subsample, colsample_bytree, gamma):
        """
        :param learning_rate:学习率
        :param n_estimators: 树个数
        :param max_depth: 每棵树最大深度
        :param min_child_weight: 最小节点权重
        :param subsample: 子采样比例
        :param colsample_bytree:列采样比例
        :param gamma:惩罚系数
        :return:-RMSE
        """
        xgb_model = xgb.XGBRegressor(learning_rate=learning_rate, n_estimators=int(n_estimators),
                                     max_depth=int(max_depth),
                                     min_child_weight=int(min_child_weight), seed=0, subsample=subsample,
                                     colsample_bytree=colsample_bytree, gamma=gamma, random_state=2
                                     ).fit(self.new_features, self.dataset[96 * 3:])
        predict_y = xgb_model.predict(self.new_features)
        RMSE = np.sqrt(mean_squared_error(self.dataset[96 * 3:], predict_y))
        return -RMSE

    def optimize_parameters(self):

        rf_bo = BayesianOptimization(
            self.rf_cv,
            {'learning_rate': (0.001, 1),
             'n_estimators': (100, 1000),
             'max_depth': (1, 20),
             'min_child_weight': (0, 10),
             'subsample': (0, 1),
             'colsample_bytree': (0, 1),
             'gamma': (0, 1)}
        )

        rf_bo.maximize()
        best_params = rf_bo.res.max["params"]
        return best_params

    def train_model(self):
        """
        :return: XGBModel
        """
        train_x = self.new_features
        train_y = list(self.dataset[96 * 3:])
        xgb_model = xgb.XGBRegressor(learning_rate=0.2975, n_estimators=610, max_depth=4, min_child_weight=2, seed=0,
                                     subsample=0.7, colsample_bytree=0.8767, gamma=0.04301, reg_alpha=1, reg_lambda=1)

        xgb_model = xgb_model.fit(train_x, train_y)
        # R2 = xgb_model.score(train_x, train_y)
        # predict_y = xgb_model.predict(train_x)
        # RMSE = np.sqrt(mean_squared_error(train_y, predict_y))
        # print("模型拟合优度为:%4f;     RMSE为：%4f" % (R2, RMSE))
        #
        # # plt.rcParams['font.sans-serif'] = ['SimHei']
        # # plt.rcParams['axes.unicode_minus'] = False
        # plt.figure(figsize=(20, 7), dpi=80)
        # plt.plot(train_y, label="原始序列")
        # plt.plot(predict_y, label="预测序列")
        # plt.title("模型训练情况")
        # plt.legend()
        # plt.show()
        return xgb_model

    def next_Node_predict(self, xgb_model, number=1):
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
        next_node_prediction = xgb_model.predict(next_node_feature)
        self.dataset.loc[tempTime] = next_node_prediction
        return (str(tempTime), next_node_prediction[0])

    def former_prediction(self, number, xgb_model):
        """
        :param number: the numbwe we want to predict
        :return: timeSeries
        """
        time = []
        prediction = []
        for i in range(number):
            one_node_prediction = self.next_Node_predict(xgb_model)
            time.append(pd.to_datetime(one_node_prediction[0]))
            prediction.append(one_node_prediction[1])
        return pd.Series(prediction, index=time)

