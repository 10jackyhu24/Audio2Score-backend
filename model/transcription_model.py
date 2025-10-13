# model/transcription_model.py
import torch
import torch.nn as nn

class TranscriptionModel(nn.Module):
    def __init__(self, num_pitch=88, hidden_size=256, num_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_pitch = num_pitch
        self.lstm = None  # input_size 會在 forward 時動態設定
        self.fc = nn.Linear(hidden_size, num_pitch)

    def forward(self, x):
        # x: [B, N_MELS, T]
        x = x.unsqueeze(1)  # [B, 1, N_MELS, T]
        x = self.cnn(x)     # [B, C, H, W]
        B, C, H, W = x.shape

        x = x.permute(0, 3, 1, 2).contiguous()  # [B, W, C, H]
        x = x.view(B, W, C*H)                   # [B, seq_len=W, input_size=C*H]

        # 動態初始化 LSTM
        if self.lstm is None:
            self.lstm = nn.LSTM(input_size=C*H,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                batch_first=True)
            if x.is_cuda:
                self.lstm = self.lstm.cuda()

        x, _ = self.lstm(x)       # [B, seq_len, hidden_size]
        x = self.fc(x)            # [B, seq_len, num_pitch]
        return x
