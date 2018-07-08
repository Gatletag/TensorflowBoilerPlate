import sys
import os
sys.path.append(os.getcwd())

import tensorflow as tf

from dataset.dataset import Dataset
from device.device import Device
from dataset.dataiterator import DataIterator

def main():
    # change to get arguments from command line
    device = Device()
    session = device.get_session()

    data_folder = "dataset/data/cifar-10"
    input_folder = data_folder+"/train"
    label_file = data_folder+"/trainLabels.csv"
    batch_size = 32

    dataset = Dataset(data_folder, input_folder, label_file, True)

    TrainingSet = DataIterator(dataset.train_x, dataset.train_y, batch_size)
    TestSet = DataIterator(dataset.test_x, dataset.test_y, batch_size)

if __name__ == '__main__':
    main()
