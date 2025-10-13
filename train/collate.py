# train/collate.py
import torch
from torch.nn.utils.rnn import pad_sequence

def pad_collate_fn(batch):
    specs, labels = zip(*batch)  # batch = list of tuples

    # 取 batch 裡最長的時間長度
    max_spec_len = max(s.shape[1] for s in specs)
    max_label_len = max(l.shape[1] for l in labels)

    # pad spectrogram
    specs_padded = [torch.nn.functional.pad(s, (0, max_spec_len - s.shape[1])) for s in specs]
    specs_tensor = torch.stack(specs_padded)

    # pad pianoroll
    labels_padded = [torch.nn.functional.pad(l, (0, max_label_len - l.shape[1]), value=-1) for l in labels]
    labels_tensor = torch.stack(labels_padded)

    return specs_tensor, labels_tensor
