# inference.py
import torch
import librosa
import numpy as np
from model.transcription_model import TranscriptionModel
from data.midi_utils import midi_to_pianoroll
from config import SAMPLE_RATE, N_MELS, HOP_LENGTH
import pretty_midi
import os
from config import *

MODEL_PATH = "./output/checkpoints/model_final.pt"

def load_model(model_path, device="cpu"):
    model = TranscriptionModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def audio_to_mel(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return torch.tensor(mel_db).float().unsqueeze(0)  # [1, 229, T]

def pianoroll_to_midi(pianoroll, output_path="output.mid", fs=100):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    (num_notes, num_frames) = pianoroll.shape
    frame_length = 1.0 / fs
    for note in range(num_notes):
        active = pianoroll[note] > 0.5
        start = None
        for t, is_active in enumerate(active):
            if is_active and start is None:
                start = t * frame_length
            elif not is_active and start is not None:
                end = t * frame_length
                pitch = note + 21
                note_obj = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
                instrument.notes.append(note_obj)
                start = None
    pm.instruments.append(instrument)
    pm.write(output_path)
    print("✅ MIDI 已輸出到:", output_path)

if __name__ == "__main__":
    os.makedirs(RESULT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(MODEL_PATH, device)

    audio_path = "./test.wav"  # 放你要測試的音檔
    mel = audio_to_mel(audio_path).to(device)

    with torch.no_grad():
        output = model(mel)  # [1, T, 88]
        pred = torch.sigmoid(output).cpu().numpy()[0]  # [T, 88]
        pred = pred.T  # [88, T] for pianoroll_to_midi
        pred = (pred >= 0.5).astype(float)

    pianoroll_to_midi(pred, output_path="./output/results/test.mid")
