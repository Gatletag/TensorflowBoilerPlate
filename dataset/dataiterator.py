import numpy as np

class DataIterator(object):
    def __init__(self, input, labels, batch_size, shuffle=False):
        assert input.shape[0] == labels.shape[0], 'Got different numbers of data and labels'
        self.input, self.labels = input, labels
        self.batch_size, self.shuffle = batch_size, shuffle
        self.total_data = self.input.shape[0]
        self.total_batches = self.total_data // self.batch_size
        self.init_iterator()

    def init_iterator(self):
        idxs = np.arange(self.total_data)
        if self.shuffle:
            np.random.shuffle(idxs)
            self.input = self.input[idxs]
            self.labels = self.labels[idxs]
        self.iterator = iter((self.input[i:i+self.batch_size], self.labels[i:i+self.batch_size]) for i in range(0, self.total_data, self.batch_size))

    def get_batch_counts(self):
        return self.total_batches

    def get_next_batch(self):
        data_x, data_y = next(self.iterator)
        return data_x, data_y

    def reset_iterator(self):
        self.iterator = iter((self.input[i:i+self.batch_size], self.labels[i:i+self.batch_size]) for i in range(0, self.total_data, self.batch_size))
