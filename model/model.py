import tensorflow as tf

class Model(object):
    def __init__(self, config):
        """
        Initiates the model process
        """
        self.config = config

        self.curr_epoch = tf.Variable(0, dtype=tf.int32, trainable=False, name="curr_epoch")
        self.increment_epoch_counter = tf.assign(self.curr_epoch, self.curr_epoch + 1)

        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

        self.saver = tf.train.Saver(max_to_keep=None)

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

    def restore_model(self, sess):
        if config["checkpoint_path"] is not None:
            checkpoint = tf.train.latest_checkpoint(config["checkpoint_path"])
            self.saver.restore(sess, checkpoint)
            print("Loaded model from latest checkpoint:", checkpoint)

    def save_model(self, sess):
        self.saver.save(sess, self.config["checkpoint_path"], self.global_step)
        print("Saved model")

    def build_network(self):
        """
        Creates Neural Network architecture
        """
        raise NotImplementedError
