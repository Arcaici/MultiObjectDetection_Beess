from collections import Counter

import pandas as pd
from matplotlib import pyplot as plt
import shutil

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from Dataset import VOCDataset
import torch
from torchvision import transforms
import xml.etree.ElementTree as ET  # For XML parsing
import os
import re
from PIL import Image
from utils import display_img, display_bbox


def parse_labels_from_xmls(xml_paths):
    """
    Parses the Pascal VOC XML annotation file and returns labels unique names and units.

    Args:
        xml_paths (str): Path to the XML annotation files.

    Returns:
        list: one list containing the class labels and the number of them.
    """
    labels = set()
    for xml in xml_paths:
        tree = ET.parse(xml)
        root = tree.getroot()

        for obj in root.findall('object'):
            name = obj.find('name').text
            labels.add(name)

    return labels
def parse_size_from_xmls(xml_paths, min_size=None):
    """
        Parses the Pascal VOC XML annotation file and returns all unique sizes.

        Args:
            xml_paths  (str): Path to the XML annotation files.
            min_size  (tuple)(width,height): minimum width and height resolution

        Returns:
            list: one list containing all sizes from labels. if min_size is not None return all sizes over these values.
        """
    sample = False
    if min_size:
        sample = True
    sizes = set()
    for xml in xml_paths:
        tree = ET.parse(xml)
        root = tree.getroot()
        size = root.find('size')
        width = size.find('width').text
        height = size.find('height').text
        item = width+"x"+height
        if sample:
            if int(width) >= min_size[0] and int(height) >= min_size[1]:
                sizes.add(item)
        else:
            sizes.add(item)
    return sizes
def parse_size_dict_from_xmls(xml_paths, sizes):
    """
            Parses the Pascal VOC XML annotation file and returns labels unique names and units.

            Args:
                xml_paths (str): Path to the XML annotation files.
                sizes     (set)(str): All sizes to count.

            Returns:
                list: one list containing all sizes from labels and the number of them.
            """
    sizes_dict = {}
    for size in sizes:
        sizes_dict[size] = 0
    for xml in xml_paths:
        tree = ET.parse(xml)
        root = tree.getroot()
        size = root.find('size')
        width = size.find('width').text
        height = size.find('height').text
        key = width + "x" + height
        if sizes_dict.get(key) is not None:
            sizes_dict[key] += 1
    return sizes_dict
def plot_dict_images_sizes(dict, title="Resolution of Images"):
    """
                Plot a bar chart that show the first 20 resolution after sorting values in descending order.

                Args:
                    dict    dict(str)(int): dict copntaining sizes and units to plot
                    title            (str): plot title

    """
    if len(dict.keys()) > 20:
        sorted_data = sorted(dict.items(), key=lambda x: x[1], reverse=True)[:20]
    else:
        sorted_data = sorted(dict.items(), key=lambda x: x[1], reverse=True)

    # Extract keys and values from the sorted data
    keys = [item[0] for item in sorted_data]
    values = [item[1] for item in sorted_data]

    # Plotting the data
    plt.figure(figsize=(8, 6))
    bars = plt.bar(keys, values)
    plt.xlabel('Resolution')
    plt.ylabel('# Images')
    plt.title(title)

    # Adding count labels above each bar
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha='center', va='bottom')


    plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()
def select_only_HD_files(xml_paths):
    """
      Retrieves all XML files That describe a image with a HD resolution or higher.

      Args:
          xml_paths (str): Path to the directory containing XML files.

      Returns:
          list: List of paths to all XML files with HD or higher resolution.
      """
    hd_files = []
    for xml in xml_paths:
        tree = ET.parse(xml)
        root = tree.getroot()
        size = root.find('size')
        width = size.find('width').text
        height = size.find('height').text
        item = width + "x" + height

        if int(width) >= 640 and int(height) >= 480:
            hd_files.append(xml)
    return hd_files
def find_image_paths(xml_paths):
    """
       Retrive from all xml files the associated image filename

              Args:
                  xml_paths (str): Path to the directory containing XML files.
              Returns:
                  image_paths list(str): contains all the images file name

    """
    image_paths=[]
    for xml in xml_paths:
        print(xml)
        tree = ET.parse(xml)
        root = tree.getroot()
        image_paths.append(root.find('filename').text)

    return image_paths
def move_selected_files(xml_paths, source_folder = "./ML-Data", destination_folder = "./Selected_data"):
    """
          Move all XML files and Images selected to Selected_data folder

          Args:
              xml_paths (str): Path to the directory containing XML files.

    """

    image_paths = find_image_paths(xml_paths)

    for path in xml_paths:
        file = os.path.basename(path)
        destination_path = os.path.join(destination_folder, file)
        shutil.copyfile(path, destination_path)
        print(f"copyfile '{file}' to '{destination_path}'")

    for path in image_paths:
        destination_path = os.path.join(destination_folder, path)
        source_path = os.path.join(source_folder,path)
        shutil.copyfile(source_path, destination_path)
        print(f"Moved '{path}' to '{destination_path}'")
def get_beehive_name(xml_paths):
    """
       Retrive from all xml files the associated beehive name

              Args:
                  xml_paths (str): Path to the directory containing XML files.
              Returns:
                  beehive dict: contains all the beehive name as keys and the occurencies as values

    """
    beehive_names=[]
    for xml in xml_paths:
        tree = ET.parse(xml)
        root = tree.getroot()
        filename = root.find('filename').text
        temp = filename.split(sep="_")
        if temp[0] == "Chueried" or temp[0] == "OldSchoolHoney":
            beehive_names.append(temp[0] + "_" + temp[1])
        else:
            beehive_names.append(temp[0]+"_"+temp[1]+"_"+temp[2])

        occurrences = Counter(beehive_names)
        beehives = dict(occurrences)

    return beehives, beehive_names
def get_xml_files(dir_path):
  """
  Retrieves all XML files from a directory and its subdirectories.

  Args:
      dir_path (str): Path to the directory containing XML files.

  Returns:
      list: List of paths to all XML files.
  """
  all_files = []
  for root, _, files in os.walk(dir_path):
    for filename in files:
      if filename.endswith(".xml"):
        file_path = os.path.join(root, filename)
        all_files.append(file_path)
  return all_files

def check_image_corruption(dir_path):
    """
           Retrive a list of all images that are corrupted

                  Args:
                      dir_path (str): Path to the directory containing Image files.
                  Returns:
                      corrupetd_images (list): contains all the images names that are corrupted
        """
    img_path_all = [f for f in os.listdir(dir_path) if f.endswith('.jpg') or f.endswith('.png')]
    corrupetd_images = []
    for img_path in img_path_all:
        try:
            full_path = os.path.join(dir_path, img_path)
            img = Image.open(full_path)
            img.verify()
        except(IOError,SyntaxError) as e:
            print("Bad image: ", img_path)
            corrupetd_images.append(img_path)
    return corrupetd_images
def remove_corrupted_img(img_path, dir_path):
    """
       Remove all images from the desire path
           Args:
               img_path (list): List of all images paths
               dir_path (str): Path to the directory containing Image files.
    """
    for img in img_path:
        full_path = os.path.join(dir_path, img)
        xml = img.replace(".jpg", ".xml")
        xml = xml.replace(".png", ".xml")
        xml_full_path = os.path.join(dir_path, xml)
        os.remove(xml_full_path)
        os.remove(full_path)
def split_and_save_dataset(xml_path):
    dir_path = "./Selected_data"
    # Test and Train
    df = pd.DataFrame(xml_path, columns=["xml"])
    proj_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    # Train and Valis + Stratify respect to hives names
    xml_proj_df = [os.path.join(dir_path, it) for it in proj_df["xml"].tolist()]
    _, proj_df["hive"]= get_beehive_name(xml_proj_df)
    train_df, val_df = train_test_split(proj_df,stratify=proj_df["hive"], test_size=0.3, random_state=42)

    # Dropping hive column
    train_df = train_df.drop(columns=['hive'])
    val_df = val_df.drop(columns= ['hive'])

    # Obtaining xml lists
    test_df = test_df['xml'].tolist()
    test_df = [os.path.join(dir_path, it) for it in test_df]
    train_df = train_df['xml'].tolist()
    train_df = [os.path.join(dir_path, it) for it in train_df]
    val_df = val_df['xml'].tolist()
    val_df = [os.path.join(dir_path, it) for it in val_df]

    move_selected_files(test_df, source_folder= dir_path, destination_folder=os.path.join(dir_path, "test"))
    move_selected_files(train_df, source_folder= dir_path, destination_folder=os.path.join(dir_path, "train"))
    move_selected_files(val_df, source_folder= dir_path, destination_folder=os.path.join(dir_path, "valid"))


# # ---------------------- DOWNLOADED DATASET ----------------------#

# path to dataset files obtained from kaggle
# dir_path = "./ML-Data"
# xml_files = get_xml_files(dir_path)
#
# # number of classes
# class_labels = parse_labels_from_xmls(xml_files)
# print(f"Class labels: {class_labels}")
#
# # All image resolution and HD image resolution
# img_sizes = parse_size_from_xmls(xml_files)
# hd_img_sizes = parse_size_from_xmls(xml_files, min_size=(640,480))
#
# img_dict_sizes = parse_size_dict_from_xmls(xml_files, img_sizes)
# hd_img_dict_sizes = parse_size_dict_from_xmls(xml_files, hd_img_sizes)
#
# print(f"img size: {img_dict_sizes}")
# print(f"hd img size: {hd_img_dict_sizes}")
#
# plot_dict_images_sizes(img_dict_sizes, title="All images resolution")
# plot_dict_images_sizes(hd_img_dict_sizes, title="All HD images and higher resolutions")
#
# # Selecting only HD data
# hd_files = select_only_HD_files(xml_files)
# print("# hd files: ",len(hd_files))
# class_labels = parse_labels_from_xmls(hd_files)
# print(f"Class labels HD: {class_labels}")
#
# move_selected_files(hd_files)
#
# #---------------------- SELECTED DATASET ----------------------#
#
# # Checking corruptedImages
# dir_path = "./Selected_data"
# corrupted = check_image_corruption(dir_path)
# print(len(corrupted))
# xml_corrupted = [it.replace(".jpg", ".xml") for it in corrupted]
# xml_corrupted = [it.replace(".png", ".xml") for it in xml_corrupted]
# xml_corrupted = [os.path.join(dir_path,it) for it in xml_corrupted]
#
# corrupted_bh, _ =get_beehive_name(xml_corrupted)
# print(corrupted_bh)
#
# # All image resolution and HD image resolution
# hd_corrupted_sizes = parse_size_from_xmls(xml_corrupted)
#
# hd_corrupted_dict_sizes = parse_size_dict_from_xmls(xml_corrupted, hd_corrupted_sizes)
# plot_dict_images_sizes(hd_corrupted_dict_sizes, title="All images resolution")
#
# remove_corrupted_img(corrupted,dir_path)
#
# #path to dataset files obtained from selected images
# dir_path = "./Selected_data"
# xml_files = get_xml_files(dir_path)
#
# hd_beehives, _ = get_beehive_name(xml_files)
# print(hd_beehives)
#
# hd_labels = parse_labels_from_xmls(xml_files)
# print(hd_labels)
#
# # All image resolution and HD image resolution
# hd_img_sizes = parse_size_from_xmls(xml_files)
#
# hd_img_dict_sizes = parse_size_dict_from_xmls(xml_files, hd_img_sizes)
# plot_dict_images_sizes(hd_img_dict_sizes, title="All images resolution")

# # ------------- SPLITTING DATASET --------------#
# dir_path = "./Selected_data"
# xml_path_all = [f for f in os.listdir(dir_path) if f.endswith('.xml')]
#
# split_and_save_dataset(xml_path_all)
# # ------------- TESTING DATASET CLASS --------------#
# img_width = 640
# img_height = 480
# image_dir = "./Selected_data"
# str2label = {"pad": -1, "bee": 0}
# label2str = {-1: "pad", 0: "bee"}
# transform = transforms.Compose([
#     transforms.ToTensor(),  # Convert PIL images to tensors
#     # ... other transformations ...
# ])
#
# dataset = VOCDataset(img_dir=image_dir, img_size=(img_height, img_width), str2label=str2label, label2str=label2str, transform=transform)
# od_dataloader = DataLoader(dataset, batch_size=2)
# for img_batch, gt_bboxes_batch, gt_classes_batch in od_dataloader:
#     img_data_all = img_batch
#     gt_bboxes_all = gt_bboxes_batch
#     gt_classes_all = gt_classes_batch
#     break
#
# img_data_all = img_data_all[:2]
# gt_bboxes_all = gt_bboxes_all[:2]
# gt_classes_all = gt_classes_all[:2]
#
# # get class names
# gt_class_1 = gt_classes_all[0].long()
# gt_class_1 = [label2str[idx.item()] for idx in gt_class_1]
#
# gt_class_2 = gt_classes_all[1].long()
# gt_class_2 = [label2str[idx.item()] for idx in gt_class_2]
#
# nrows, ncols = (1, 2)
# fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
#
# fig, axes = display_img(img_data_all, fig, axes)
# fig, _ = display_bbox(gt_bboxes_all[0], fig, axes[0], classes=gt_class_1)
# fig, _ = display_bbox(gt_bboxes_all[1], fig, axes[1], classes=gt_class_2)
#
# plt.show()