import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
class GetFeatures(object):
    def __init__(self,dataset):
        """
        :param dataset: we use original time_series dataset to create features
        """
        self.dataset = dataset
    def get_next_node_feature(self,i = 1):
        """
        :return: features with shape of (,106)
        """
        next_node_feature = []
        temp_node = self.dataset[-i]
        seasonal_feature = [self.dataset[-96-i],self.dataset[-96*2-i],self.dataset[-96*3-i]]
        t_range_feature = self.dataset[-96*1-i:-i]
        t_info_feature = [np.array(t_range_feature).mean(),np.array(t_range_feature).min(),
                          np.array(t_range_feature).max(),np.array(t_range_feature).std()]
        temp_index = self.dataset.index[-i]
        date_feature = [temp_index.day,temp_index.hour,temp_index.weekday()]
        next_node_feature.extend(seasonal_feature)
        next_node_feature.extend(t_range_feature)
        next_node_feature.extend(t_info_feature)
        next_node_feature.extend(date_feature)
        return next_node_feature

    def get_last_number_feature(self,number):
        """
        :param number: we get the last amount of dataset to create features
        :return: features with shape of (number,106)
        """
        lastNumber_node_features = []
        for i in range(1,number+1):
            last_node_feature = self.get_next_node_feature(i)
            lastNumber_node_features.append(last_node_feature)
#             self.dataset.pop(self.dataset.shape[0]-1)
        return np.array(lastNumber_node_features)

    def get_all_seasonal_features(self):
        """
        :return: get all seasonal features with shape of (dataet.shape[0]-96*3,3)
        """
        seasonal_3 = self.dataset.values[:self.dataset.shape[0] - 96 * 3]
        seasonal_2 = self.dataset.values[96:self.dataset.shape[0] - 96 * 2]
        seasonal_1 = self.dataset.values[96 * 2:self.dataset.shape[0] - 96]
        all_seasonal_feature = np.array([seasonal_3,seasonal_2,seasonal_1]).T
        return all_seasonal_feature

    def get_all_trange_features(self):
        """
        :return: get all seasonal features with shape of (dataset.shape[0]-96*3,96)
        """
        all_trange_features = []
        data = list(self.dataset)
        for i in range(96 * 3, self.dataset.shape[0]):
            t_range = data[i - 96:i]
            all_trange_features.append(data[i - 96:i])
        return np.array(all_trange_features)

    def get_all_tinfo_features(self):
        """
        :return: get all seasonal features with shape of (dataset.shape[0]-96*3,4)
        """
        all_t_min = []
        all_t_max = []
        all_t_mean = []
        all_t_std = []
        data = list(self.dataset)
        for i in range(96 * 3, self.dataset.shape[0]):
            t_range = data[i - 96:i]
            t_mean = np.array(t_range).mean()
            t_min = np.array(t_range).min()
            t_max = np.array(t_range).max()
            t_std = np.array(t_range).std()
            all_t_max.append(t_max)
            all_t_min.append(t_min)
            all_t_mean.append(t_mean)
            all_t_std.append(t_std)
        all_tinfo_features = np.array([all_t_min,all_t_max,all_t_mean,all_t_std]).T
        return all_tinfo_features

    def get_all_date_features(self):
        """
        :return: get all date features with shape of (dataset.shape[0]-96*3,3)
        """
        all_date_features = []
        for i in range(96 * 3, self.dataset.shape[0]):
            all_date_features.append([self.dataset.index[i].day,self.dataset.index[i].hour,self.dataset.index[i].weekday()])
        return np.array(all_date_features)

    def get_original_features(self):
        """
        :return: get all features with shape of (dataset.shape[0]-96*3,106)
        """
        all_features = []
        all_seasonal_features = self.get_all_seasonal_features()
        all_trange_features = self.get_all_trange_features()
        all_tinfo_features = self.get_all_tinfo_features()
        all_date_features = self.get_all_date_features()
        all_features = np.hstack((all_seasonal_features,all_trange_features))
        all_features = np.hstack((all_features,all_tinfo_features))
        all_features = np.hstack((all_features,all_date_features))
        print("数据特征为原始时间序列的第288个之后的数据的特征")
        return all_features
