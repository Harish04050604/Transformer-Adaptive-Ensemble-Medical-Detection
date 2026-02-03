import torch
from torch.utils.data import DataLoader
from dataset import PolypDataset
from lrpm_ham_model import LRPM_HAM_Detector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = PolypDataset("labelled/img", "labelled/yolo")

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)

model = LRPM_HAM_Detector().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 3   

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    print(f"\nStarting Epoch {epoch+1}/{EPOCHS}")

    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Progress print every 5 images
        if i % 5 == 0:
            print(f"  Batch {i}/{len(loader)} | Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} finished | Total Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "lrpm_ham_detector.pth")
print("\n LRPM-HAM model trained and saved as lrpm_ham_detector.pth")
