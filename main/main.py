import sys
import os
sys.path.append(os.getcwd())

import tensorflow as tf

from dataset.dataset import Dataset
from device.device import Device
from dataset.dataiterator import DataIterator
from model.custom.testmodel import TestModel
from trainer.train import Train
from logger.logger import Logger
from utils.utils import create_directories
from trainer.inference import Inference

def main():
    # change to get arguments from command line
    data_folder = "dataset/data/cifar-10"
    input_folder = data_folder+"/train"
    label_file = data_folder+"/trainLabels.csv"
    batch_size = 128

    dataset = Dataset(data_folder, input_folder, label_file, True)

    TrainingSet = DataIterator(dataset.train_x, dataset.train_y, batch_size)
    TestSet = DataIterator(dataset.test_x, dataset.test_y, batch_size)

    config = {
        "learning_rate":0.001,
        "other":1,
        "checkpoint_path":"checkpoint/test",
        "checkpoint_name":"test",
        "summary_path":"graphs/test",
        "training": False,
        "epochs": 55
    }
    try:
        create_directories(["checkpoint", "graphs"])
    except:
        print("Cannot create directory")
    tf.reset_default_graph()



    model = TestModel(config)
    model.build_model()

    device = Device()
    session = device.get_session()

    session.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    if config["training"]:
        model.restore_model(saver, session)

        logger = Logger(session, config)
        trainer = Train(session, model, saver, TrainingSet, TestSet, logger)

        trainer.train(config["epochs"])
    else:
        inference = Inference(session, model, config, saver)
        inference.load_from_saved_model(saver)

        prediction = inference.prediction(TestSet)
        print(prediction)
if __name__ == '__main__':
    main()
