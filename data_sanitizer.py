import sys, os, shutil
from pathlib import Path

INPUT_DATA_FOLDER = './data/unsanitized'
OUTPUT_DATA_FOLDER = './data/sanitized'

IMAGE_SIZE = 640
invalid_boxes = 0

def line_to_bbox(line):
    """
    Converts a line of normalized coordinates
    to a bounding box.
    """
    parts = list(map(float, line.strip().split()))
    coords = parts[1:]
    x_coords = coords[0::2]
    y_coords = coords[1::2]

    # Convert polygon to bounding box
    xmin = min(x_coords) * IMAGE_SIZE
    ymin = min(y_coords) * IMAGE_SIZE
    xmax = max(x_coords) * IMAGE_SIZE
    ymax = max(y_coords) * IMAGE_SIZE
    
    height = ymax - ymin
    width = xmax - xmin

    if height <= 0 or width <= 0:
        print('Invalid bbox, Skipping: ', line)
        global invalid_boxes
        invalid_boxes += 1
        return None

    return [xmin, ymin, xmax, ymax]

def sanitize_directory(data_type_dir):
    root = os.path.join(INPUT_DATA_FOLDER, data_type_dir)
    print(f'Entering {root}')

    all_in_folder = lambda subdir: list(sorted(os.listdir(os.path.join(root, subdir))))
    images_raw = all_in_folder('images') 
    labels_raw = all_in_folder('labels')

    names = []
    images_out = []
    bboxes_out = []

    for image_path, label_path in zip(images_raw, labels_raw):
        label_path = os.path.join(root, 'labels', label_path)
        image_path = os.path.join(root, 'images', image_path)

        name = Path(image_path).stem
        bboxes = []

        with open(label_path, 'r') as label_file:
            for line in label_file.readlines():
                bbox = line_to_bbox(line)

                if bbox != None:
                    bboxes.append(bbox)

        if len(bboxes) > 0:
            images_out.append(image_path)
            bboxes_out.append(bboxes)
            names.append(name)
    
    output_dir = os.path.join(OUTPUT_DATA_FOLDER, data_type_dir)
    image_output_path_root = os.path.join(output_dir, 'images')
    bboxes_output_path_root = os.path.join(output_dir, 'labels')

    os.makedirs(image_output_path_root, exist_ok=True)
    os.makedirs(bboxes_output_path_root, exist_ok=True) 
    print(f'Writing to {output_dir}')

    for image_path, bboxes, name in zip(images_out, bboxes_out, names):
        image_output_path = os.path.join(image_output_path_root, f'{name}.png')
        bbox_output_path = os.path.join(bboxes_output_path_root, f'{name}.txt')

        shutil.copy(image_path, image_output_path)
        with open(bbox_output_path, 'w') as bbox_output_file:
            for bbox in bboxes:
                print(bbox[0], bbox[1], bbox[2], bbox[3], 
                      file=bbox_output_file) 

def main():
    sanitize_directory('train')
    sanitize_directory('test')
    sanitize_directory('valid')
    print(f'Invalid bounding boxes: {invalid_boxes}')

if __name__ == '__main__':
    main()
