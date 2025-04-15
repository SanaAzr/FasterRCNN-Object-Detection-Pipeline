import torch
import os
import torchvision.transforms.functional as F
import cv2
import numpy as np
from Initialize import model, device
from datasets import test_loader

# Function to calculate IoU


def calculate_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union by taking the intersection
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

# Function to evaluate the model on the test dataset and calculate simple accuracy


def evaluate_model(model, test_loader, device, iou_threshold=0.5):
    model.eval()
    true_positives = 0
    total_ground_truths = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for target, output in zip(targets, outputs):
                gt_boxes = target['boxes'].cpu().numpy()
                pred_boxes = output['boxes'].cpu().numpy()

                for gt_box in gt_boxes:
                    total_ground_truths += 1
                    for pred_box in pred_boxes:
                        iou = calculate_iou(pred_box, gt_box)
                        if iou >= iou_threshold:
                            true_positives += 1
                            break

    accuracy = true_positives / total_ground_truths if total_ground_truths else 0
    return accuracy


# Load the trained model weights
model.load_state_dict(torch.load("final_model.pth", map_location=device))
model.to(device)

# Evaluate the model
accuracy = evaluate_model(model, test_loader, device)
print(f"Simple IoU-based Accuracy: {accuracy}")

# Function to save images with detected bounding boxes


def save_detected_images(test_loader, model, device, output_dir='./detected_images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.eval()
    for i, (images, targets) in enumerate(test_loader):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for j, image in enumerate(images):
            np_image = image.permute(1, 2, 0).cpu().numpy()
            np_image = (np_image * 255).astype(np.uint8)
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

            pred_boxes = outputs[j]['boxes'].data.cpu().numpy()
            for box in pred_boxes:
                start_point = (int(box[0]), int(box[1]))
                end_point = (int(box[2]), int(box[3]))
                color = (0, 255, 0)
                thickness = 2
                np_image = cv2.rectangle(
                    np_image, start_point, end_point, color, thickness)

            cv2.imwrite(f"{output_dir}/image_{i}_{j}.jpg", np_image)


# Save detected images
save_detected_images(test_loader, model, device)
