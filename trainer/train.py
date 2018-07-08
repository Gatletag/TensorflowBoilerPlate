import tensorflow as tf
import numpy as np

class Train(object):
    def __init__(self, sess, model, saver, dataset, logger):
        self.sess = sess
        self.model = model
        self.dataset = dataset
        self.logger = logger
        self.saver = saver
        # self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # self.sess.run(self.init)
        # try:
        #     self.model.restore_model(self.sess)
        # except Exception:
        #     print("No previous model to restore from.")


    def train(self, total_epochs):
        saved_epoch_for_model = self.model.curr_epoch.eval(self.sess)
        global_step = self.model.global_step.eval(self.sess)

        for epoch in range(saved_epoch_for_model, total_epochs):
            self.train_epoch()
            self.sess.run(self.model.increment_epoch_counter)

    def train_epoch(self):
        losses = []
        accuracies = []

        # loop for batches
        for i in range(self.dataset.get_batch_counts()):
            loss, accuracy = self.train_step()
            losses.append(loss)
            accuracies.append(accuracy)

        mean_loss = np.mean(losses)
        std_loss = np.std(losses)

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        curr_iteration = self.model.global_step.eval(self.sess)
        curr_epoch = self.model.curr_epoch.eval(self.sess)
        print("Loss for " + str(curr_epoch) + ": " + str(mean_loss) +" +/- " + str(std_loss))
        print("Accuracy for " + str(curr_epoch) + ": " + str(mean_accuracy) +" +/- " + str(std_accuracy))
        # TODO: add weights to summaries and gradients (?)

        summaries = {
            "loss": mean_loss,
            "accuracy": mean_accuracy
        }

        self.logger.add_to_summary(curr_iteration, curr_epoch, summaries)
        self.model.save_model(self.sess, self.saver,curr_epoch)
        # self.saver.save(self.sess, "checkpoints/convnet_layers/mnist-convnet", curr_epoch)

        self.dataset.reset_iterator()

    def train_step(self):
        batch_x, batch_y = self.dataset.get_next_batch()
        # feed_dict = {self.model.input: batch_x, self.model.labels: batch_y, self.model.isTraining: True}
        feed_dict = {self.model.input: batch_x, self.model.labels: batch_y}
        _, loss, accuracy =  self.sess.run([self.model.opt, self.model.loss, self.model.accuracy], feed_dict=feed_dict)
        return loss, accuracy
