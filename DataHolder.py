import numpy as np


class DataHolder:
    def __init__(self):
        self.input_ids = np.load('DBD/input_ids01.npy')
        self.input_ids2 = np.load('DBD/input_ids02.npy')
        self.input_segments = np.load('DBD/input_segments01.npy')
        self.input_segments2 = np.load('DBD/input_segments02.npy')
        #input_ids_3이 대체 뭔가
        input_ids = np.load('DBD/input_ids_3.npy')
        input_ids2 = np.load('DBD/input_ids_n_3.npy')
        input_segments = np.load('DBD/input_segments_3.npy')
        input_segments2 = np.load('DBD/input_segments_n_3.npy')

        self.input_ids = np.concatenate([self.input_ids, input_ids], axis=0)
        self.input_ids2 = np.concatenate([self.input_ids2, input_ids2], axis=0)
        self.input_segments = np.concatenate([self.input_segments, input_segments], axis=0)
        self.input_segments2 = np.concatenate([self.input_segments2, input_segments2], axis=0)

        self.batch_size = 12

        self.ix = 0
        self.tx = 0

        #4000->0 2개
        self.r_ix = np.array(range(4000, self.input_ids.shape[0]), dtype=np.int32)
        np.random.shuffle(self.r_ix)

    def next_batch(self):
        if self.ix + self.batch_size + 4000 >= self.input_ids.shape[0]:
            self.ix = 0

        input_ids = np.zeros(shape=[self.batch_size, 128], dtype=np.int32)
        input_segments = np.zeros(shape=[self.batch_size, 128], dtype=np.int32)
        dialog_label = np.zeros(shape=[self.batch_size, 2], dtype=np.float32)

        for i in range(int(self.batch_size / 2)):
            ix = self.r_ix[self.ix]

            input_ids[i] = self.input_ids[ix]
            input_ids[i + int(self.batch_size / 2)] = self.input_ids2[ix]

            input_segments[i] = self.input_segments[ix]
            input_segments[i + int(self.batch_size / 2)] = self.input_segments2[ix]

            dialog_label[i, 0] = 1                                  #[1, 0]
            dialog_label[i + int(self.batch_size / 2), 1] = 1       #[0, 1]

            self.ix += 1
        #print(input_ids)
        #print('---')

        return input_ids, input_segments, dialog_label

    def test_batch(self):
        if self.ix + self.batch_size >= self.input_ids.shape[0]:
            self.ix = 1000

        input_ids = np.zeros(shape=[2, 128], dtype=np.int32)
        input_segments = np.zeros(shape=[2, 128], dtype=np.int32)
        dialog_label = np.zeros(shape=[2, 2], dtype=np.int32)

        for i in range(1):
            input_ids[i] = self.input_ids[self.tx]
            input_ids[i + 1] = self.input_ids2[self.tx]

            input_segments[i] = self.input_segments[self.tx]
            input_segments[i + 1] = self.input_segments2[self.tx]

            dialog_label[i, 0] = 1
            dialog_label[i + 1, 1] = 1

            self.tx += 1

        return input_ids, input_segments, dialog_label



