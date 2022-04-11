import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from joblib import dump
# from icecream import ic
import numpy as np
import math
import matplotlib as mpl
from matplotlib.image import imread
from random import randint
from tensorflow import keras
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import optimizers
import tensorflow.keras.utils
import tensorflow.keras.layers
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
import copy
import csv

class SensorDataset(Dataset):


    def normalize_data(self,dataset, data_min, data_max):
        data_std = (dataset - data_min) / (data_max - data_min)
        test_scaled = data_std * (np.amax(data_std) - np.amin(data_std)) + np.amin(data_std)
        return test_scaled


    def import_data(self,train_dataframe, dev_dataframe, test_dataframe):
        dataset = train_dataframe.values
        dataset = dataset.astype('float32')

        #Include all 12 initial factors (Year ; Month ; Hour ; Day ; Cloud Coverage ; Visibility ; Temperature ; Dew Point ;
        #Relative Humidity ; Wind Speed ; Station Pressure ; Altimeter
        max_test = np.max(dataset[:,12])
        min_test = np.min(dataset[:,12])
        scale_factor = max_test - min_test
        max = np.empty(13)
        min = np.empty(13)

        #Create training dataset
        for i in range(0,13):
            min[i] = np.amin(dataset[:,i],axis = 0)
            max[i] = np.amax(dataset[:,i],axis = 0)
            dataset[:,i] = self.normalize_data(dataset[:, i], min[i], max[i])

        train_data = dataset[:,0:12]
        train_labels = dataset[:,12]

        # Create dev dataset
        dataset = dev_dataframe.values
        dataset = dataset.astype('float32')

        for i in range(0, 13):
            dataset[:, i] = self.normalize_data(dataset[:, i], min[i], max[i])

        dev_data = dataset[:,0:12]
        dev_labels = dataset[:,12]

        # Create test dataset
        dataset = test_dataframe.values
        dataset = dataset.astype('float32')

        for i in range(0, 13):
            dataset[:, i] = self.normalize_data(dataset[:, i], min[i], max[i])

        test_data = dataset[:, 0:12]
        test_labels = dataset[:, 12]

        return train_data, train_labels, dev_data, dev_labels, test_data, test_labels, scale_factor
    """Face Landmarks dataset."""

    def __init__(self,value):

        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory
        """
        self.transform = MinMaxScaler()
        self.data_pv = pd.read_csv('data/pv-generation-dataset.csv')
        data_except_night = self.data_pv[self.data_pv['Time'].isin(range(6,18))]
        train_df, dev_df, test_df = np.split(data_except_night.sample(frac=1, random_state=32), [int(.8*len(data_except_night)), int(.9*len(data_except_night))])
        self.train_data, self.train_labels, self.dev_data, self.dev_labels, self.test_data, self.test_labels, self.scale_factor = self.import_data(train_df, dev_df, test_df)
        self.value = value
    def __len__(self):
        # return number of sensors
        return len(self.data_pv)

    # Will pull an index between 0 and __len__. 
    def __getitem__(self,time_steps):
        
        # Sensors are indexed from 1
        
        time_steps = 1
        assert(self.train_data.shape[0] % time_steps == 0)
        
        X_train = torch.tensor(np.reshape(self.train_data, (self.train_data.shape[0] // time_steps, time_steps, self.train_data.shape[1])))
        X_dev = torch.tensor(np.reshape(self.dev_data, (self.dev_data.shape[0] // time_steps, time_steps, self.dev_data.shape[1])))
        X_test = torch.tensor(np.reshape(self.test_data, (self.test_data.shape[0] // time_steps, time_steps, self.test_data.shape[1])))
        Y_train = torch.tensor(np.reshape(self.train_labels, (self.train_labels.shape[0] // time_steps, time_steps, 1)))
        Y_dev = torch.tensor(np.reshape(self.dev_labels, (self.dev_labels.shape[0] // time_steps, time_steps, 1)))
        Y_test = torch.tensor(np.reshape(self.test_labels, (self.test_labels.shape[0] // time_steps, time_steps, 1)))

        # print(X_train.shape,"_______X_train____")
        # print(Y_train.shape,"_____Y_train___")
        # print(X_test.shape,"_____X_test______")
        # print(Y_test.shape,"_____Y_test____")
        # print(X_dev.shape,"_____X_dev_____")
        # print(Y_dev.shape,"____Y_dev_____")
        
        scaler_feature = self.transform
        scaler_target = self.transform

        scaler_feature.fit(X_train[:,0,0].unsqueeze(-1))
        scaler_target.fit(Y_train[:,0,0].unsqueeze(-1))


        X_train[:,0,0] = torch.tensor(scaler_feature.transform(X_train[:,0,0].unsqueeze(-1)).squeeze(-1).astype(np.float32))
        X_dev[:,0,0] = torch.tensor(scaler_feature.transform(X_dev[:,0,0].unsqueeze(-1)).squeeze(-1).astype(np.float32))
        X_test[:,0,0] = torch.tensor(scaler_feature.transform(X_test[:,0,0].unsqueeze(-1)).squeeze(-1).astype(np.float32))

        Y_train[:,0,0] = torch.tensor(scaler_target.transform(Y_train[:,0,0].unsqueeze(-1)).squeeze(-1).astype(np.float32))
        Y_dev[:,0,0] = torch.tensor(scaler_target.transform(Y_dev[:,0,0].unsqueeze(-1)).squeeze(-1).astype(np.float32))
        Y_test[:,0,0] = torch.tensor(scaler_target.transform(Y_test[:,0,0].unsqueeze(-1)).squeeze(-1).astype(np.float32))

        # print(X_train)
        # input("!!!!!")



        # print(X_train.shape,"_______X_train____")
        # print(Y_train.shape,"_____Y_train___")
        # print(X_test.shape,"_____X_test______")
        # print(Y_test.shape,"_____Y_test____")
        # print(X_dev.shape,"_____X_dev_____")
        # print(Y_dev.shape,"____Y_dev_____")
        # input("!!!!!")

        dump(scaler_feature, 'scalar_feature.joblib')
        dump(scaler_target, 'scalar_target.joblib')
        if self.value == "train":
            return X_train , Y_train
        if self.value == "dev":
            return  X_dev , Y_dev
        if self.value == "test":
            return X_test, Y_test 


