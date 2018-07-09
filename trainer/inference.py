import tensorflow as tf
import numpy as np

class Inference(object):
    def __init__(self,sess, model, config, saver):
        self.sess = sess
        self.model = model
        self.config = config

    def load_from_saved_model(self, saver):
        self.model.restore_model(saver, self.sess)

    def prediction(self, inputs):
        results = []
        for i in range(inputs.get_batch_counts()):
            result = self.predict(inputs)
            results.append(result)
        results = np.array(results)
        results = results.reshape(-1, results.shape[-1])

        inputs.reset_iterator()

        return np.argmax(results,axis=1)

    def predict(self,inputs):
        batch_x, batch_y = inputs.get_next_batch()
        feed_dict = {self.model.input: batch_x, self.model.isTraining: False}
        batch_results =  self.sess.run([self.model.prediction], feed_dict=feed_dict)
        return batch_results
