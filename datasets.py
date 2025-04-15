import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import transforms
import json


class CustomCocoDataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transforms=None):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.imgs = []
        self.annotations = []

        # Temporarily store annotation data to check for emptiness
        temp_annotations = [file for file in sorted(
            os.listdir(annotation_dir)) if file.endswith('.json')]

        # Only add images and annotations if the annotation file is not empty
        for ann_file in temp_annotations:
            ann_path = os.path.join(self.annotation_dir, ann_file)
            with open(ann_path) as f:
                annotations = json.load(f)
                # Check if annotations list is not empty
                if annotations:
                    self.annotations.append(ann_file)
                    img_file = ann_file.replace('.json', '.jpg')
                    if os.path.exists(os.path.join(self.img_dir, img_file)):
                        self.imgs.append(img_file)
                    else:
                        print(
                            f"Image file {img_file} corresponding to annotations {ann_file} does not exist.")

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = read_image(img_path)
        img = img.float() / 255.0

        ann_path = os.path.join(self.annotation_dir, self.annotations[idx])
        with open(ann_path) as f:
            annotations = json.load(f)

        if not annotations:
            print(f"No annotations found for {self.annotations[idx]}")
            return torch.zeros(1, 1, 1, 1), {}

        boxes = []
        labels = []
        image_id = annotations[0]['image_id']
        for ann in annotations:
            x_min, y_min, width, height = ann['bbox']
            boxes.append([x_min, y_min, x_min + width, y_min + height])
            labels.append(ann['category_id'])

        # Convert everything into a PyTorch tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(
            (0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros(
            (0,), dtype=torch.int64)
        # Define the target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id])
        }

        # Apply the transformations
        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


# Define the transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((800, 800)),
    transforms.ToTensor()
])

# Create the dataset objects
train_dataset = CustomCocoDataset(
    './datasets/train', './annotations/train', transforms=transform)
val_dataset = CustomCocoDataset(
    './datasets/val', './annotations/val', transforms=transform)
test_dataset = CustomCocoDataset(
    './datasets/test', './annotations/test', transforms=transform)

# Create the data loader objects
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
