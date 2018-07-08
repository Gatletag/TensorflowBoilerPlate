import tensorflow as tf
from model.model import Model

class TestModel(Model):
    def __init__(self, config):
        super(TestModel, self).__init__(config)

    def build_network(self):
        """
        Creates Neural Network architecture
        """
        self.input = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.labels = tf.placeholder(tf.int32, [None,10])

        initializer = tf.variance_scaling_initializer(scale=2.0)
        conv1 = tf.layers.Conv2D(filters=20,kernel_size=(3,3),padding='same',
                             data_format='channels_last',use_bias=True,activation=tf.nn.relu,kernel_initializer=initializer)
        conv2 = tf.layers.Conv2D(filters=20,kernel_size=(3,3),padding='same',
                             data_format='channels_last',use_bias=True,activation=tf.nn.relu,kernel_initializer=initializer)
        mp1 = tf.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid')
        conv3 = tf.layers.Conv2D(filters=16,kernel_size=(3,3),padding='same',
                             data_format='channels_last',use_bias=True,activation=tf.nn.relu,kernel_initializer=initializer)
        conv4 = tf.layers.Conv2D(filters=16,kernel_size=(3,3),padding='same',
                             data_format='channels_last',use_bias=True,activation=tf.nn.relu,kernel_initializer=initializer)
        mp2 = tf.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid')
        conv5 = tf.layers.Conv2D(filters=16,kernel_size=(3,3),padding='same',
                             data_format='channels_last',use_bias=True,activation=tf.nn.relu,kernel_initializer=initializer)
        ft = tf.layers.Flatten()
        fc1 = tf.layers.Dense(64,kernel_initializer=initializer)
        fc2 = tf.layers.Dense(10,kernel_initializer=initializer)
        layer = [conv1,conv2,mp1,conv3,conv4,mp2,conv5,ft,fc1,fc2]#

        model = tf.keras.Sequential(layer)
        # self.logits = tf.layers.dense(fc1, 10)
        self.logits = model(self.input)

    def loss(self):
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name='loss')

    def optimize(self):
        self.opt = tf.train.AdamOptimizer(self.config["learning_rate"]).minimize(self.loss, global_step=self.global_step)

    def evaluate(self):
        with tf.name_scope('predict'):
            y_pred = tf.argmax(self.logits, 1)
            labels = tf.argmax(self.labels, 1)
            correct_preds = tf.equal(y_pred, labels)
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
