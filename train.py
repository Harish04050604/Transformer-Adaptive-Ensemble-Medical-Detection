#train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from transformer_model import PolypSegformer
from segmentation_dataset import PolypDataset
from loss import BCEDiceLoss

IMAGE_DIR   = "labelled/img"
MASK_DIR    = "labelled_masks"
TRAIN_SPLIT = "train_split.txt"
VAL_SPLIT   = "val_split.txt"
SAVE_PATH   = "transformer_segformer.pth"
BATCH_SIZE  = 4
EPOCHS      = 20
LR          = 6e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_ds     = PolypDataset(IMAGE_DIR, MASK_DIR, TRAIN_SPLIT, augment=True)
val_ds       = PolypDataset(IMAGE_DIR, MASK_DIR, VAL_SPLIT,   augment=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

model     = PolypSegformer().to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = BCEDiceLoss()

best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            loss = criterion(model(images), masks)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)
    scheduler.step()

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✔ Best model saved (val loss: {val_loss:.4f})")

print(f"\nTraining complete. Best model saved to: {SAVE_PATH}")