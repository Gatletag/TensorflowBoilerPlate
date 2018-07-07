import tensorflow as tf

class Model(object):
    def __init__(self, save_path=None, sess=None, **kwargs):
        """
        Initiates the model process
        """
        self.lr = kwargs["learning_rate"]
        self.batch_size = kwargs["batch_size"]
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        self.isTraining = kwargs["isTraining"]

    def build_model(self):
        """
        Sets up the computation graph
        """
        self.inference()
        self.loss()
        self.optimize()
        self.evaluate()
        self.summary()

    def loss(self):
        """
        Defines the loss function
        """
        with tf.name_scope('loss'):
            self.loss = None

    def optimize(self):
        """
        Defines the optimizer function
        """
        self.optimizer = None #tf.train.AdamOptimizer(self.lr).minize(self.loss, global_step=self.gstep)

    def evaluate(self):
        """
        Evaluates the accuracy of the network
        """
        with tf.name_scope("accuracy"):
            self.accuracy = None

    def inference(self,input,output):
        """
        Creates the neural network architecture
        (Optionally define the architecture in another file)
        """
        self.input = tf.placeholder()
        self.true_output = tf.placeholder()
        pass

    def restore_model(self, save_path):
        if save_path is not None:
            saver = tf.train.Saver()
            saver.restore(sess, save_path)
