import tensorflow as tf
import os

class Logger(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.summary_vals = {}
        self.summary_ops = {}


        self.writer_train = tf.summary.FileWriter(os.path.join(self.config["summary_path"], "train"), self.sess.graph)
        self.writer_test = tf.summary.FileWriter(os.path.join(self.config["summary_path"], "test"))

    def add_to_summary(self, iter_step, epoch_step, summaries_dict=None, type="train"):
        if type == "train":
            summary_writer = self.writer_train
        else:
            summary_writer = self.writer_test

        with tf.name_scope('summaries'):
            if summaries_dict != None:
                summary_list = []
                for key, value in summaries_dict.items():
                    if key not in self.summary_ops:
                        if len(value.shape) <= 1:
                            self.summary_vals[key] = tf.placeholder('float32', value.shape, name=key)
                            self.summary_ops[key] = tf.summary.scalar(key, self.summary_vals[key])
                        else:
                            self.summary_vals[key] = tf.placeholder('float32', [None] +list(value.shape[1:]), name=key)
                            self.summary_ops[key] = tf.summary.image(key, self.summary_vals[key])

                    summary_list.append(self.sess.run(self.summary_ops[key],feed_dict={self.summary_vals[key]:value}))

                for summary in summary_list:
                    summary_writer.add_summary(summary, epoch_step)
                summary_writer.flush()
