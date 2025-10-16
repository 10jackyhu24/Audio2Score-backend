# AudioConversionTrain.py
import torch
from torch.utils.data import DataLoader
from train.collate import pad_collate_fn
from config import *
from data.dataset_loader import MaestroDataset
from model.transcription_model import TranscriptionModel
from train.trainer import train_model
import os

if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset = MaestroDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)

    model = TranscriptionModel()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()

    trained_model = train_model(model, dataloader, optimizer, criterion, device, epochs=EPOCHS)
    torch.save(trained_model.state_dict(), os.path.join(CHECKPOINT_DIR, "model_final.pt"))
    print("✅ 模型已保存至:", CHECKPOINT_DIR)
