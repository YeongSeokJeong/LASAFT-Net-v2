from torch.utils.data import Dataset
from abc import ABCMeta
from pathlib import Path
import random
import numpy as np

from hydra.utils import to_absolute_path
from lasaft.utils.functions import audioset_label_indices
from glob import glob
import soundfile as sf
from os.path import exists


class AudiosetWrapper(Dataset):
    __metaclass__ = ABCMeta

    def __init__(self, audioset_root, subset, n_fft, hop_length, num_frame):

        audioset_root = Path(to_absolute_path(audioset_root))
        meta_path = f'{audioset_root}/meta_data'.replace('\\', '/')
        self.window_length = hop_length * (num_frame - 1)
        if subset == 'train':
            self.wav_dict, self.label_dict, self.lengths = self.get_paths(f'{audioset_root}/data/balanced_train_segments'.replace('\\', '/'))
        else:
            self.wav_dict, self.label_dict, self.lengths = self.get_paths(f'{audioset_root}/data/eval_segments'.replace('\\', '/'))
        self.id2idx, self.id2name = audioset_label_indices(meta_path)

        self.num_tracks = len(self.wav_dict)
        self.num_targets = len(self.id2idx)

    def get_paths(self, path):
        wav_paths = []
        lengths = []
        for wav_path in glob(f'{path}/*.wav'):
            length = sf.read(wav_path)[0].shape[0]
            if exists(wav_path[:-3] + 'txt') and length > self.window_length:
                wav_paths.append(wav_path)
                lengths.append(length)
        label_paths = [wav_path[:-3] + 'txt' for wav_path in wav_paths]
        return wav_paths, label_paths, lengths

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def get_audio(self, idx, pos=0, length=None):
        arg_dicts = {
            'file': self.wav_dict[idx],
            'start': pos,
            'dtype': 'float32'
        }

        if length is not None:
            arg_dicts['stop'] = pos + length

        return sf.read(**arg_dicts)[0]

    def get_label(self, idx):
        with open(self.label_dict[idx], 'r') as f:
            idx = [self.id2idx[id.strip()] for id in f.readline().strip().split(',')]
        return idx

    def to_multi_hot(self, label):
        condition = np.zeros(len(self.id2idx))
        condition[label] = 1.
        condition /= sum(condition)
        return condition.astype(np.float32)


class AudiosetTrainDataset(AudiosetWrapper):
    def __init__(self, audioset_root, n_fft, hop_length, num_frame):
        super().__init__(audioset_root, 'train', n_fft, hop_length, num_frame)

    def __len__(self) -> int:
        return sum([length//self.window_length for length in self.lengths])

    def __getitem__(self, whatever):
        idx1 = random.randint(0, self.num_tracks - 1)
        wav1, id1 = self.get_random_audio(idx1), self.get_label(idx1)

        while True:
            idx2 = random.randint(0, self.num_tracks - 1)
            wav2, id2 = self.get_random_audio(idx2), self.get_label(idx2)
            if set(id1).isdisjoint(id2):
                break
        mixture = wav1 + wav2

        return mixture, wav1, wav2, self.to_multi_hot(id1), self.to_multi_hot(id2)

    def get_random_audio(self, idx):
        max_len = self.lengths[idx] - self.window_length - 1
        start_pos = random.randint(0, max_len)
        return self.get_audio(idx, start_pos, self.window_length)

