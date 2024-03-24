import os
from torchvision import ops
import numpy as np
import torch
import xml.etree.ElementTree as ET  # For XML parsing
import matplotlib.patches as patches

#--------- XML Tools ---------#
def parse_annotation(image_dir, image_path_all, img_size):
    """
            Parses the Pascal VOC XML annotation files of all images and returns bounding boxes and labels.

            Args:
                image_dir           (str):  Path to the image files.
                image_data_all     (list): All images paths.
                img_size        (int,int): The image resize dimensions

            Returns:
                list, list: Two lists, the first containing all images bounding boxes and
                            the second containing all corresponding class labels.
    """

    img_h, img_w = img_size
    gt_boxes_all = []
    gt_classes_all = []

    for img_path in image_path_all:
        xml_path = os.path.join(image_dir, os.path.splitext(img_path)[0] + '.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # get raw image size
        size = root.find("size")
        orig_w = int(size.find("width").text)
        orig_h = int(size.find("height").text)

        # get bboxes and their labels
        groundtruth_boxes = []
        groundtruth_classes = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # rescale bboxes
            bbox = torch.Tensor([xmin, ymin, xmax, ymax])
            bbox[[0, 2]] = bbox[[0, 2]] * img_w / orig_w
            bbox[[1, 3]] = bbox[[1, 3]] * img_h / orig_h

            groundtruth_boxes.append(bbox.tolist())

            groundtruth_classes.append(label)

        gt_boxes_all.append(torch.Tensor(groundtruth_boxes))
        gt_classes_all.append(groundtruth_classes)

    return gt_boxes_all, gt_classes_all

#--------- Visualization Tools ------------#
def display_img(img_data, fig, axes):
    for i, img in enumerate(img_data):
        if type(img) == torch.Tensor:
            img = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img)

    return fig, axes
def display_bbox(bboxes, fig, ax, classes=None, in_format='xyxy', color='y', line_width=3):
    if type(bboxes) == np.ndarray:
        bboxes = torch.from_numpy(bboxes)
    if classes:
        assert len(bboxes) == len(classes)
    # convert boxes to xywh format
    bboxes = ops.box_convert(bboxes, in_fmt=in_format, out_fmt='xywh')
    c = 0
    for box in bboxes:
        x, y, w, h = box.numpy()
        # display bounding box
        rect = patches.Rectangle((x, y), w, h, linewidth=line_width, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        # display category
        if classes:
            if classes[c] == 'pad':
                continue
            ax.text(x + 5, y + 20, classes[c], bbox=dict(facecolor='yellow', alpha=0.5))
        c += 1

    return fig, ax