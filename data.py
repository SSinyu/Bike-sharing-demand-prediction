import numpy as np
from tensorflow.keras import utils


class DataGenerator(utils.Sequence):
    def __init__(self, data, mode="train", input_width=72, target_width=1, batch_size=64, n_station=128):
        super(DataGenerator, self).__init__()
        self.mode = mode
        self.batch_size = batch_size
        self.n_station = n_station

        self.x, self.y = self.get_subset(data, input_width, target_width, n_station)

        self.indexes = np.arange(len(self.x))
        self.input_size = self.x[0].shape
        self.target_size = self.y[0].shape
        self.on_epoch_end()
        
    def __len__(self):
        return len(self.y) // self.batch_size

    def on_epoch_end(self):
        if self.mode == "train":
            np.random.shuffle(self.indexes)

    def __getitem__(self, i):
        _indexes = self.indexes[i*self.batch_size:(i+1)*self.batch_size]
        batch_x = np.zeros((self.batch_size, *self.input_size), dtype=np.float32)
        batch_y = np.zeros((self.batch_size, *self.target_size), dtype=np.float32)
        for i, ind in enumerate(_indexes):
            batch_x[i] = self.x[ind]
            batch_y[i] = self.y[ind]
        return batch_x, batch_y

    @staticmethod
    def get_subset(data, input_width=72, target_width=1, n_station=128):
        total_width = input_width + target_width
        interv = len(data)//n_station

        x, y = [], []
        for stat_i in range(0, len(data), interv):
            d_stat = data[stat_i:(stat_i+interv), :]
            for i in range(0, len(d_stat)-total_width):
                d = d_stat[i:(i+total_width), :]
                x.append(d[:input_width, :])
                y.append(np.squeeze(d[input_width:, :2]))
        return x, y
