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

        conv1 = tf.layers.conv2d(inputs=self.input,
                                  filters=32,
                                  kernel_size=[3, 3],
                                  padding='SAME',
                                  # activation=tf.nn.relu,
                                  name='conv1')
        bnconv1 = tf.layers.batch_normalization(conv1,training=self.isTraining)
        reluconv1 = tf.nn.relu(bnconv1);
        pool1 = tf.layers.max_pooling2d(inputs=reluconv1,
                                        pool_size=[2, 2],
                                        strides=2,
                                        name='pool1')

        conv2 = tf.layers.conv2d(inputs=pool1,
                                  filters=64,
                                  kernel_size=[3, 3],
                                  padding='SAME',
                                  # activation=tf.nn.relu,
                                  name='conv2')
        bnconv2 = tf.layers.batch_normalization(conv2,training=self.isTraining)
        reluconv2 = tf.nn.relu(bnconv2);
        pool2 = tf.layers.max_pooling2d(inputs=reluconv2,
                                        pool_size=[2, 2],
                                        strides=2,
                                        name='pool2')

        feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
        pool2 = tf.reshape(pool2, [-1, feature_dim])
        fc = tf.layers.dense(pool2, 1024, activation=tf.nn.relu, name='fc')
        dropout = tf.layers.dropout(fc,
                                    0.5,
                                    training=self.isTraining,
                                    name='dropout')
        self.logits = tf.layers.dense(dropout, 10, name='logits')

        self.prediction = tf.nn.softmax(self.logits)

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
            self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
