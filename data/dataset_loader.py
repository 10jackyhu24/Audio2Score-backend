# data/dataset_loader.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
from .midi_utils import midi_to_pianoroll
from config import SAMPLE_RATE, CLIP_SECONDS, N_MELS, HOP_LENGTH

class MaestroDataset(Dataset):
    def __init__(self, data_dir):
        self.data_pairs = []
        self.clip_len = CLIP_SECONDS * SAMPLE_RATE
        self.time_steps = int(self.clip_len / HOP_LENGTH) + 1

        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith(".wav"):
                    midi_file = f.replace(".wav", ".midi")
                    midi_path = os.path.join(root, midi_file)
                    wav_path = os.path.join(root, f)
                    if os.path.exists(midi_path):
                        self.data_pairs.append((wav_path, midi_path))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        wav_path, midi_path = self.data_pairs[idx]

        # load audio
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
        if len(y) > self.clip_len:
            start = np.random.randint(0, len(y) - self.clip_len)
            y = y[start:start+self.clip_len]
        else:
            y = np.pad(y, (0, self.clip_len - len(y)))

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # load MIDI
        pianoroll = midi_to_pianoroll(midi_path, fs=100)
        # 裁切或 pad 到 time_steps
        if pianoroll.shape[1] > self.time_steps:
            pianoroll = pianoroll[:, :self.time_steps]
        else:
            pad_width = self.time_steps - pianoroll.shape[1]
            pianoroll = np.pad(pianoroll, ((0,0), (0,pad_width)))

        return torch.tensor(mel_db).float(), torch.tensor(pianoroll).float()
