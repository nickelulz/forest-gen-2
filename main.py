import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN

from engine import train_one_epoch, evaluate
from util import *
from dataset import * 

def main() -> None:
    device = set_device()
    dataset = load_dataset('./data/forest')
    model = get_model_instance_segmentation(num_classes = 2)
    train_model(model, dataset, device, num_epochs=2)
    torch.save(model.state_dict(), 'forest-stand-gen.pth')
    test_model(model, "data/forest/test/images/stand_1809.png")

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


def test_model(model, image_path) -> None:
    """
    Trains the model on one image (not actual "testing" occurs
    implicitly in train_model).
    """
    image = read_image(image_path)
    eval_transform = get_transform(train=False)


    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]


    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    masks = (pred["masks"] > 0.7).squeeze(1)
    output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))

if __name__ == '__main__':
    main()
