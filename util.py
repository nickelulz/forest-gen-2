import os, random

import cv2
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T

def get_model_object_detection(num_classes):
    # Load a Faster R-CNN model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained box predictor with a new one for the given number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

def set_device():
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda') 
    else: 
        device = torch.device('cpu')

    print(f'Using Device: {device}')

def visualize_bboxes(folder_path, num_images=5, img_size=640):
    """
    Visualize bounding polygons (stands) on randomly selected images.
    
    folder_path: Path to the parent folder which has 'images/' and 'labels/' subfolders.
    num_images: Number of random images to visualize.
    img_size:   The width/height of the square images (default: 640).
    """

    images_dir = os.path.join(folder_path, 'images')
    labels_dir = os.path.join(folder_path, 'labels')

    # List all image filenames in images_dir
    all_image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith('.png')
    ]

    # Choose random subset of images
    chosen_images = random.sample(all_image_files, min(num_images, len(all_image_files)))

    # Create a figure with subplots
    fig, axes = plt.subplots(1, len(chosen_images), figsize=(5 * len(chosen_images), 5))
    if len(chosen_images) == 1:
        axes = [axes]  # make it iterable if there's only one image

    for ax, image_file in zip(axes, chosen_images):
        # Read the image
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)
        
        # Convert from BGR (OpenCV) to RGB (matplotlib)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Construct the label filename (same base name but .txt)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)

        # Plot the image
        ax.imshow(image)
        ax.set_title(image_file)
        ax.axis('off')

        # If there's a corresponding label file, read and plot polygons
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                data = line.strip().split()
                
                # Typically data[0] is the class (in your case '0')
                class_id = data[0]
                
                # The rest are x1,y1,x2,y2,x3,y3,x4,y4,x1,y1 (normalized coords)
                coords = [float(x) for x in data[1:]]
                
                # Convert normalized coordinates into pixel values
                coords = np.array(coords) * img_size  # multiply each by 640
                
                # Reshape into (N_points, 2). 
                # For a box polygon with 5 repeated points, you get shape (5, 2)
                coords = coords.reshape(-1, 2)

                # Create a polygon patch
                polygon = patches.Polygon(
                    coords,
                    closed=True,
                    fill=False,            # Don't fill the polygon
                    edgecolor='red',
                    linewidth=2
                )
                ax.add_patch(polygon)

    plt.tight_layout()
