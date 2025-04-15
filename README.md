# Faster R-CNN Object Detection Project

This project implements an end-to-end object detection pipeline using PyTorch and OpenCV. It leverages a Faster R-CNN model fine-tuned on a custom dataset (a subset inspired by the COCO format) to detect objects in images. The project covers data preprocessing, model training, evaluation, and inference using a pretrained model.

## Project Overview

The pipeline is organized as follows:

1. **Data Preparation and Splitting**

   - **splitting_dataset.py**  
     Splits the raw images and annotations into training, validation, and test sets (70% / 15% / 15% respectively).

2. **Dataset Management**

   - **datasets.py**  
     Defines a custom dataset class that reads images and their COCO-style annotations. It also sets up data loaders for training, validation, and testing with appropriate transformations.

3. **Model Initialization**

   - **Initialize.py**  
     Loads a pre-trained Faster R-CNN model and replaces its prediction head to suit the number of classes. It also sets up the optimizer and device configuration.

4. **Model Training**

   - **Train.py**  
     Implements the training loop over multiple epochs. The code saves intermediate checkpoints and the final model for later use.

5. **Validation and Evaluation**

   - **Validation.py**  
     Evaluates the model on the validation set by computing average loss over the batches.
   - **Test.py**  
     Evaluates the model on the test set using an Intersection over Union (IoU)-based accuracy metric. Additionally, it saves images annotated with predicted bounding boxes.

6. **Inference with a Pre-trained Model**
   - **Pretrained_object_Detection.py**  
     Demonstrates how to run inference using a pre-trained Faster R-CNN, applying non-maximum suppression (NMS) and drawing bounding boxes on input images.

## How to Run the Project

1. **Environment Setup:**

   - Ensure you have Python 3.x installed.
   - Install required libraries: PyTorch, torchvision, OpenCV, Matplotlib, and other dependencies.

2. **Prepare the Data:**

   - Place your dataset (images and corresponding COCO-style annotations) into the appropriate folders.
   - Run `splitting_dataset.py` to organize the dataset into `train`, `val`, and `test` splits.

3. **Training the Model:**

   - Execute `Train.py` to start training the model. Intermediate checkpoints and the final model (`final_model.pth`) will be saved.

4. **Validation and Testing:**

   - Run `Validation.py` to evaluate the model on the validation set.
   - Execute `Test.py` to calculate a simple IoU-based accuracy on the test set and save images with detected objects.

5. **Running Inference:**
   - Use `Pretrained_object_Detection.py` for running inference with a pre-trained model to compare results or for quick demos.

## Future Improvements

- Experiment with additional training epochs and fine-tuning of hyperparameters.
- Enhance the evaluation metrics by incorporating more comprehensive metrics such as mAP.
- Optimize data augmentation strategies to further boost model performance.

## Dependencies

- Python 3.x
- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [OpenCV](https://opencv.org/)
- [Matplotlib](https://matplotlib.org/)
