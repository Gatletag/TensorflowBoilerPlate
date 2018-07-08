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

def main():
    # change to get arguments from command line
    # device = Device()
    # session = device.get_session()

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
        "checkpoint_path":"checkpoint_files",
        "checkpoint_folder":"test",
        "summary_path":"graphs"
    }
    tf.reset_default_graph()

    model = TestModel(config)
    model.build_model()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        model.restore_model(saver, session)
        # ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_layers/checkpoint'))
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(session, ckpt.model_checkpoint_path)
    #
        logger = Logger(session, config)
    #
    #
        trainer = Train(session, model, saver, TrainingSet, logger)
    #
    # print(model.global_step.eval(session), model.curr_epoch.eval(session))
    #
        trainer.train(55)

if __name__ == '__main__':
    main()
