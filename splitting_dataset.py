import os
import shutil
import numpy as np

images_path = './pictures'
annotations_path = './annotations'

train_dir = 'train'
val_dir = 'val'
test_dir = 'test'

for split_dir in [train_dir, val_dir, test_dir]:
    os.makedirs(os.path.join(images_path, split_dir), exist_ok=True)
    os.makedirs(os.path.join(annotations_path, split_dir), exist_ok=True)

all_images = os.listdir(images_path)
all_annotations = os.listdir(annotations_path)

# Shuffle the list for random splits
np.random.seed(42)
np.random.shuffle(all_images)

# Define split sizes (70% train, 15% val, 15% test)
num_images = len(all_images)
train_end = int(num_images * 0.7)
val_end = int(num_images * 0.85)

# Split the images
train_images = all_images[:train_end]
val_images = all_images[train_end:val_end]
test_images = all_images[val_end:]

# Function to move files to their respective directories


def move_files(file_list, source_dir, split_dir):
    for file_name in file_list:
        source_file = os.path.join(source_dir, file_name)
        target_dir = os.path.join(source_dir, split_dir)
        if os.path.isfile(source_file):
            shutil.move(source_file, target_dir)


# Move the files
move_files(train_images, images_path, train_dir)
move_files(val_images, images_path, val_dir)
move_files(test_images, images_path, test_dir)

train_annotations = [name.replace('.jpg', '.json') for name in train_images]
val_annotations = [name.replace('.jpg', '.json') for name in val_images]
test_annotations = [name.replace('.jpg', '.json') for name in test_images]

move_files(train_annotations, annotations_path, train_dir)
move_files(val_annotations, annotations_path, val_dir)
move_files(test_annotations, annotations_path, test_dir)

print("Files have been moved to their respective directories.")
