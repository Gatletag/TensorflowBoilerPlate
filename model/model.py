import tensorflow as tf
from utils.utils import create_directories
import os

class Model(object):
    def __init__(self, config):
        """
        Initiates the model process
        """
        self.config = config

        self.curr_epoch = tf.Variable(0, dtype=tf.int32, trainable=False, name="curr_epoch")
        self.increment_epoch_counter = tf.assign(self.curr_epoch, self.curr_epoch + 1)

        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        self.isTraining = tf.placeholder(tf.bool, name='is_training')

    def build_model(self):
        """
        Sets up the computation graph
        """
        self.build_network()
        self.loss()
        self.optimize()
        self.evaluate()

    def loss(self):
        """
        Defines the loss function
        """
        raise NotImplementedError

    def optimize(self):
        """
        Defines the optimizer function
        """
        raise NotImplementedError

    def evaluate(self):
        """
        Evaluates the accuracy of the network
        """
        raise NotImplementedError

    def restore_model(self, saver, sess):
        if self.config["checkpoint_path"] is not None:
            try:
                checkpt_loc = self.config["checkpoint_path"]+"/checkpoint"
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpt_loc))
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Loaded model from latest checkpoint:", ckpt)
            except:
                print("No Previous Model")

    def save_model(self, sess, saver, step):
        # create_directories([self.config["checkpoint_path"]])
        saver.save(sess, self.config["checkpoint_path"]+"/"+self.config["checkpoint_name"], step)
        print("Saved model")

    def build_network(self):
        """
        Creates Neural Network architecture
        """
        raise NotImplementedError
