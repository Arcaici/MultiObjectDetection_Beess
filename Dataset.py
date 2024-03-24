from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms  # For image transformations (optional)
from skimage import io
from skimage.transform import resize
from utils import parse_annotation
import os
import torch



class VOCDataset(torch.utils.data.Dataset):

    def __init__(self, img_dir, img_size, str2label, label2str, transform=None):
        """
        Args:
            img_dir   (str): Path to the dataset root directory.
            img_size  (int,int): Height and width for resize all images
            str2label (dict): Contains (classes names, label)
            label2str  (dict): Contains (label, classes names)
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.img_dir = img_dir
        self.img_size = img_size
        self.str2label = str2label
        self.label2str = label2str
        self.transform = transform
        self.img_path_all = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.img_data_all, self.gt_bboxes_all, self.gt_classes_all = self.get_data()
    def __len__(self):
        return self.img_data_all.size(dim=0)

    def __getitem__(self, idx):
        return self.img_data_all[idx], self.gt_bboxes_all[idx], self.gt_classes_all[idx]

    def get_data(self):
        img_data_all = []
        gt_idxs_all = []

        gt_boxes_all, gt_classes_all = parse_annotation(self.img_dir, self.img_path_all, self.img_size)

        for i, img_path in enumerate(self.img_path_all):

            # read and resize image
            source_path = os.path.join(self.img_dir, img_path)
            img = io.imread(source_path)
            img = resize(img, self.img_size)

            # convert image to torch tensor and reshape it so channels come first
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)

            # encode class names as integers
            gt_classes = gt_classes_all[i]
            gt_idx = torch.Tensor([self.str2label[name] for name in gt_classes])

            img_data_all.append(img_tensor)
            gt_idxs_all.append(gt_idx)
            print(f"Finita img {i}")

        # pad bounding boxes and classes so they are of the same size
        gt_bboxes_pad = pad_sequence(gt_boxes_all, batch_first=True, padding_value=-1)
        gt_classes_pad = pad_sequence(gt_idxs_all, batch_first=True, padding_value=-1)

        # stack all images
        img_data_stacked = torch.stack(img_data_all, dim=0)

        return img_data_stacked.to(dtype=torch.float32), gt_bboxes_pad, gt_classes_pad
