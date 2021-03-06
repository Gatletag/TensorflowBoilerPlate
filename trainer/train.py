import tensorflow as tf
import numpy as np

class Train(object):
    def __init__(self, sess, model, saver, train_dataset, test_dataset, logger):
        self.sess = sess
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.logger = logger
        self.saver = saver

    def train(self, total_epochs):
        saved_epoch_for_model = self.model.curr_epoch.eval(self.sess)
        global_step = self.model.global_step.eval(self.sess)

        for epoch in range(saved_epoch_for_model, total_epochs):
            self.train_epoch()
            self.test_epoch()
            self.sess.run(self.model.increment_epoch_counter)

    def train_epoch(self):
        losses = []
        accuracies = []

        # loop for batches
        for i in range(self.train_dataset.get_batch_counts()):
            loss, accuracy = self.train_step()
            losses.append(loss)
            accuracies.append(accuracy)
            if i%20 == 0:
                print(str(i)+": "+str(accuracy))

        mean_loss = np.mean(losses)
        std_loss = np.std(losses)

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        curr_iteration = self.model.global_step.eval(self.sess)
        curr_epoch = self.model.curr_epoch.eval(self.sess)
        print("[TRAIN] Loss for " + str(curr_epoch) + ": " + str(mean_loss) +" +/- " + str(std_loss))
        print("[TRAIN] Accuracy for " + str(curr_epoch) + ": " + str(mean_accuracy) +" +/- " + str(std_accuracy))

        summaries = {
            "loss": mean_loss,
            "accuracy": mean_accuracy
        }

        self.logger.add_to_summary(curr_iteration, curr_epoch, summaries)
        self.model.save_model(self.sess, self.saver,curr_epoch)

        self.train_dataset.reset_iterator()

    def train_step(self):
        batch_x, batch_y = self.train_dataset.get_next_batch()
        feed_dict = {self.model.input: batch_x, self.model.labels: batch_y, self.model.isTraining: True}
        _, loss, accuracy =  self.sess.run([self.model.opt, self.model.loss, self.model.accuracy], feed_dict=feed_dict)
        return loss, accuracy

    def test_epoch(self):
        accuracies = []

        # loop for batches
        for i in range(self.test_dataset.get_batch_counts()):
            accuracy = self.test_step()
            accuracies.append(accuracy)

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        curr_iteration = self.model.global_step.eval(self.sess)
        curr_epoch = self.model.curr_epoch.eval(self.sess)
        print("[TEST] Accuracy for " + str(curr_epoch) + ": " + str(mean_accuracy) +" +/- " + str(std_accuracy))

        summaries = {
            "accuracy": mean_accuracy
        }

        self.logger.add_to_summary(curr_iteration, curr_epoch, summaries, type="test")

        self.test_dataset.reset_iterator()

    def test_step(self):
        batch_x, batch_y = self.test_dataset.get_next_batch()
        feed_dict = {self.model.input: batch_x, self.model.labels: batch_y, self.model.isTraining: False}
        _, accuracy =  self.sess.run([self.model.logits, self.model.accuracy], feed_dict=feed_dict)
        return accuracy
