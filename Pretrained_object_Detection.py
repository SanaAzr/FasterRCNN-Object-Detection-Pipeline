import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import os

# Load the model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Function to transform the image


def transform_image(image):
    image = F.to_tensor(image)
    return image

# Function to draw bounding boxes


def draw_boxes(image, boxes, labels):
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red")
        draw.text((box[0], box[1]), str(label), fill="red")
    return image


def apply_nms(orig_prediction, iou_thresh=0.3, score_thresh=0.6):
    # Filter out predictions with low scores
    keep = orig_prediction['scores'] > score_thresh
    final_prediction = {k: v[keep].cpu() for k, v in orig_prediction.items()}

    # Apply non-maximum suppression to filter out overlapping boxes
    keep_boxes = torch.ops.torchvision.nms(
        final_prediction['boxes'], final_prediction['scores'], iou_thresh)

    final_prediction = {k: v[keep_boxes] for k, v in final_prediction.items()}
    return final_prediction


output_folder_path = './detected_objects_pretrained'

# Main inference loop


def main_inference(folder_path, num_images=100):
    # Get all files in the folder
    all_files = sorted(os.listdir(folder_path))
    # Filter out non-image files (e.g., JSON files)
    image_files = [f for f in all_files if f.lower().endswith(
        ('.png', '.jpg', '.jpeg'))]

    # Limit the number of images to process
    image_files = image_files[:num_images]

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
            # Ensure image is on the correct device
            image_tensor = transform_image(image).to(device)

            with torch.no_grad():
                # Get the model prediction
                prediction = model([image_tensor])[0]

                # Apply NMS and thresholding
                prediction = apply_nms(prediction)

            # Get boxes and labels from the prediction
            boxes = prediction['boxes']
            labels = prediction['labels']

            # Draw the boxes on the image
            image_with_boxes = draw_boxes(image, boxes, labels)

            # Save the image with bounding boxes
            output_image_path = os.path.join(output_folder_path, img_file)
            image_with_boxes.save(output_image_path)

        except Exception as e:
            print(f"An error occurred with file {img_file}: {e}")


# Specify the folder where your images are
image_folder_path = './datasets'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Run the inference
main_inference(image_folder_path)
