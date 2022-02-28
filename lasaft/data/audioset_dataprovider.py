import time
from warnings import warn

from torch.utils.data import DataLoader

from lasaft.data.audioset_wrapper import AudiosetTrainDataset


class DataProvider(object):

    def __init__(self, audioset_root,
                 batch_size, num_workers, pin_memory, n_fft, hop_length, num_frame,
                 multi_source_training, multi_condition=False, level=-1):
        self.audioset_root = audioset_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_frame = num_frame
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.multi_source_training = multi_source_training
        self.multi_condition = multi_condition
        self.level = level

    def get_training_dataset_and_loader(self):

        training_set = AudiosetTrainDataset(self.audioset_root, self.n_fft, self.hop_length, self.num_frame, self.level)

        batch_size = self.batch_size//4 if self.multi_source_training else self.batch_size
        loader = DataLoader(training_set, shuffle=True, batch_size=batch_size,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory)

        return training_set, loader

    def get_validation_dataset_and_loader(self):
        raise NotImplementedError

    def get_test_dataset_and_loader(self):
        raise NotImplementedError
