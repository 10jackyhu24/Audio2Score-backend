# train/trainer.py
import torch
from tqdm import tqdm

# 使用 BCEWithLogitsLoss 取代 CrossEntropy
def train_model(model, dataloader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for spec, label in loop:
            spec, label = spec.to(device), label.to(device)  # [B, 229, T], [B, 88, T]

            optimizer.zero_grad()
            output = model(spec)  # [B, seq_len, 88]

            # 調整 label 形狀
            label = label.permute(0, 2, 1)  # [B, T, 88]，跟 output 對齊

            # BCEWithLogitsLoss 可直接接受 multi-hot label
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())
    return model
