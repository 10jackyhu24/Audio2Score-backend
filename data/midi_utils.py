# data/midi_utils.py
import pretty_midi
import numpy as np

def midi_to_pianoroll(midi_path, fs=100):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        roll = midi_data.get_piano_roll(fs=fs)
        roll = np.clip(roll / 127.0, 0, 1)
        # 限制 pitch 範圍到 88 鍵 (A0–C8)
        roll = roll[21:109, :]
    except Exception:
        roll = np.zeros((88, 1000))
    return roll
