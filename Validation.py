import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from datasets import val_loader
from Initialize import model, device

model.load_state_dict(torch.load("final_model.pth", map_location=device))

model.to(device)
model.train()

total_loss = 0.0
num_batches = 0

with torch.no_grad():
    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        total_loss += losses.item()
        num_batches += 1

        print(f"Batch loss: {losses.item()}")

model.eval()

if num_batches > 0:
    avg_loss = total_loss / num_batches
    print(f"Average validation loss: {avg_loss}")
else:
    print("No batches were processed.")
