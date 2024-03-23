from collections import Counter

from matplotlib import pyplot as plt
import shutil
from Dataset import VOCDataset
import torch
from torchvision import transforms
import xml.etree.ElementTree as ET  # For XML parsing
import os
import re

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

        if int(width) >= 1280 and int(height) >= 720:
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
        tree = ET.parse(xml)
        root = tree.getroot()
        image_paths.append(root.find('filename').text)

    return image_paths
def move_selected_files(xml_paths):
    """
          Move all XML files and Images selected to Selected_data folder

          Args:
              xml_paths (str): Path to the directory containing XML files.

    """
    image_paths = find_image_paths(xml_paths)
    source_folder = "./ML-Data"
    destination_folder = "./Selected_data"

    for path in xml_paths:
        file = os.path.basename(path)
        destination_path = os.path.join(destination_folder, file)
        shutil.move(path, destination_path)
        print(f"Moved '{file}' to '{destination_path}'")

    for path in image_paths:
        destination_path = os.path.join(destination_folder, path)
        source_path = os.path.join(source_folder,path)
        shutil.move(source_path, destination_path)
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

    return beehives
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

# ---------------------- DOWNLOADED DATASET ----------------------#

# path to dataset files obtained from kaggle
# dir_path = "./ML-Data"
# xml_files = get_xml_files(dir_path)

# number of classes
# class_labels = parse_labels_from_xmls(xml_files)
# print(f"Class labels: {class_labels}")

# All image resolution and HD image resolution
# img_sizes = parse_size_from_xmls(xml_files)
# hd_img_sizes = parse_size_from_xmls(xml_files, min_size=(1280,720))
#
# img_dict_sizes = parse_size_dict_from_xmls(xml_files, img_sizes)
# hd_img_dict_sizes = parse_size_dict_from_xmls(xml_files, hd_img_sizes)
#
# print(f"img size: {img_dict_sizes}")
# print(f"hd img size: {hd_img_dict_sizes}")
#
# plot_dict_images_sizes(img_dict_sizes, title="All images resolution")
# plot_dict_images_sizes(hd_img_dict_sizes, title="All HD images and higher resolutions")

# Selecting only HD data
# hd_files = select_only_HD_files(xml_files)
# print(hd_files)
# print(len(hd_files))
# move_selected_files(hd_files)

# ---------------------- SELECTED DATASET ----------------------#

# path to dataset files obtained from selected images
dir_path = "./Selected_data"
xml_files = get_xml_files(dir_path)

hd_beehives = get_beehive_name(xml_files)
print(hd_beehives)

hd_labels = parse_labels_from_xmls(xml_files)
print(hd_labels)

# All image resolution and HD image resolution
hd_img_sizes = parse_size_from_xmls(xml_files)

hd_img_dict_sizes = parse_size_dict_from_xmls(xml_files, hd_img_sizes)
plot_dict_images_sizes(hd_img_dict_sizes, title="All images resolution")


# transform = transforms.Compose([
#     transforms.ToTensor(),  # Convert PIL images to tensors
#     # ... other transformations ...
# ])
#
# dataset = VOCDataset(root_dir="./ML-DATA", transform=transform)
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
#
#
# # Iterate through batches
# for images, boxes, labels in data_loader:
#     # Your training or evaluation logic here
#     print(images.shape, boxes.shape, labels.shape)
