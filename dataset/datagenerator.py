import numpy as np

class DataGenerator(object):
    def __init__(self, input, labels, batch_size, shuffle=False):
        assert input.shape[0] == labels.shape[0], 'Got different numbers of data and labels'
        self.input, self.labels = input, labels
        self.batch_size, self.shuffle = batch_size, shuffle
        self.total_data = self.input.shape[0]
        self.batch_index = 0
        self.total_batches = (int)(self.total_data / self.batch_size)

    def next_batch(self):
        if self.shuffle:
            idx = np.random.choice(self.total_data, self.batch_size)
        else:
            idx = range(self.batch_index, self.batch_index+self.batch_size , 1)
        yield self.input[idx], self.labels[idx]

    def reset_batch(self):
        self.batch_index = 0

    def get_batch_counts(self):
        return self.total_batches
