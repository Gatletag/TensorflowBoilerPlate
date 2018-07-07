import numpy as np

class DataIterator(object):
    def __init__(self, input, labels, batch_size, shuffle=False):
        assert input.shape[0] == labels.shape[0], 'Got different numbers of data and labels'
        self.input, self.labels = input, labels
        self.batch_size, self.shuffle = batch_size, shuffle
        self.total_data = self.input.shape[0]

    def __iter__(self):
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
            self.input = self.input[idxs]
            self.labels = self.labels[idxs]
        return iter((self.input[i:i+B], self.labels[i:i+B]) for i in range(0, self.total_data, self.batch_size))
