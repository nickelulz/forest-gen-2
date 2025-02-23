import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN

import matplotlib.pyplot as plt
import cv2
from PIL import Image

from engine import train_one_epoch, evaluate
from util import *
from dataset import * 

DATA_ROOT_DIR = './data/sanitized'
MODEL_TEST_IMAGE_PATH = 'test/images/stand_1809.png'
MODEL_TEST_LABEL_PATH = 'test/labels/stand_1809.txt'
MODEL_OUTPUT_FILENAME = 'forest-stand-gen.pth'

def main() -> None:
    device = set_device()
    dataset = load_dataset(DATA_ROOT_DIR, limit=1, batch_size=5)
    model = get_model_object_detection(num_classes=2)
    train_model(model, dataset, device, num_epochs=2)
    save_model(model, MODEL_OUTPUT_FILENAME)
    test_model(model, device,
               os.path.join(DATA_ROOT_DIR, MODEL_TEST_IMAGE_PATH),
               os.path.join(DATA_ROOT_DIR, MODEL_TEST_LABEL_PATH))

def save_model(model, filename):
    os.mkdirs('./bin', exist_ok=True)
    output_path = os.path.join('./bin', filename)
    torch.save(model.state_dict(), output_path)

def train_model(model, data, device, num_epochs):
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1)
    
    model.to(device)

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data.train, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data.test, device=device)

def test_model(model, device, image_path, true_label_path) -> None:
    """
    Runs the model on a single image, detects objects, and visualizes the predictions.

    Args:
        model: The trained Faster R-CNN model.
        device: The computation device (CPU or CUDA).
        image_path: Path to the image file.
    """
    # Load and preprocess the image
    model.eval()
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).to(device)

    # Run inference
    with torch.no_grad():
        predictions = model([image_tensor])

    # Convert image to OpenCV format for visualization
    image_cv = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Extract predictions
    boxes = predictions[0]["boxes"].cpu().numpy()  # Bounding boxes
    scores = predictions[0]["scores"].cpu().numpy()  # Confidence scores
    labels = predictions[0]["labels"].cpu().numpy()  # Class labels

    # Set a confidence threshold
    threshold = 0.5
    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold:
            x_min, y_min, x_max, y_max = box.astype(int)

            # Use a more readable font
            text = f"{score:.2f}"
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.5
            font_thickness = 1  # Thicker text for better clarity

            # Calculate text size
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x, text_y = x_min, y_min + text_size[1] + 2  # Position inside the bounding box

            # Draw a white background for text inside the bounding box
            cv2.rectangle(image_cv,
                          (x_min, y_min),
                          (x_min + text_size[0] + 4, y_min + text_size[1] + 4),
                          (255, 255, 255), -1)

            # Put text inside the bounding box
            cv2.putText(image_cv, text, (text_x, text_y), font,
                        font_scale, (0, 0, 0), font_thickness)

            cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)  # Black rectangle, thinner

    with open(true_label_path) as true_labels_file:
      for line in true_labels_file.readlines():
        bbox = list(map(float, line.strip().split()))
        bbox = [int(n) for n in bbox]
        cv2.rectangle(image_cv, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)

    # Display the image with bounding boxes
    plt.figure(figsize=(8, 8))
    plt.imshow(image_cv)
    plt.axis("off")
    plt.show()

if __name__ == '__main__':
    main()
