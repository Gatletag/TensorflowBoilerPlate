import numpy as np
import os
import scipy as sc
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
'''
CIFAR-10 Example
'''
class Dataset(object):
    def __init__(self, data_folder,  input_folder, label_file, shuffle = False):
        self.data_folder = data_folder
        try:
            self.load_dataset()
        except:
            self.labels = pd.read_csv(label_file, ).values
            self.labels =self.labels[1:]
            onehot_labels = self.one_hot_encode()
            self.data = np.column_stack((self.labels[:,0], onehot_labels))
            self.input_folder = input_folder

            if shuffle:
                self.shuffle_set()

            train, test = self.split_set()
            self.train_x, self.train_y = self.load_input(train)
            self.test_x, self.test_y = self.load_input(test)
            self.save_dataset()

    def shuffle_set(self):
        idxs = np.arange(self.data.shape[0])
        np.random.shuffle(idxs)
        self.data = self.data[idxs]

    def split_set(self, split_percentage = 0.8):
        train = self.data[0:int(split_percentage*self.data.shape[0])]
        test = self.data[int(split_percentage*self.data.shape[0]):]
        return train, test

    def load_input(self, dataset):
        data = []
        for file in dataset[:,0]:
            data.append(sc.ndimage.imread(self.input_folder+"/"+str(file)+".png"))
        return np.array(data), dataset[:,1]

    def one_hot_encode(self):
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(self.labels[:,1])
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        self.classes = label_encoder.classes_
        return onehot_encoded

    def save_dataset(self):
        np.save(self.data_folder+"train_x.npy", self.train_x)
        np.save(self.data_folder+"train_y.npy", self.train_y)
        np.save(self.data_folder+"test_x.npy", self.test_x)
        np.save(self.data_folder+"test_y.npy", self.test_y)

    def load_dataset(self):
        self.train_x = np.load(self.data_folder+"train_x.npy")
        self.train_y = np.load(self.data_folder+"train_y.npy")
        self.test_x = np.load(self.data_folder+"test_x.npy")
        self.test_y = np.load(self.data_folder+"test_y.npy")
