import torch
from datasets import train_loader
from Initialize import model, optimizer, device
from tqdm import tqdm


def save_model(model, path):
    torch.save(model.state_dict(), path)


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0
    for images, targets in data_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
    return total_loss / len(data_loader)


num_epochs = 30
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch}...")
    train_loss = train_one_epoch(model, optimizer, tqdm(train_loader), device)
    print(f"Epoch {epoch}, Training loss: {train_loss}")

    if epoch % 5 == 0:
        save_model(model, f"model_epoch_{epoch}.pth")

# save the final model
save_model(model, "final_model.pth")
