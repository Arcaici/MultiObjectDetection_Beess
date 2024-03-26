import os.path

import torch
import torchvision
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset import VOCDataset
from models import TwoStageDetector
from utils import display_img, project_bboxes, display_bbox

def training_loop(model, learning_rate, train_dataloader, n_epochs):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    loss_list = []

    for i in tqdm(range(n_epochs)):
        total_loss = 0
        for img_batch, gt_bboxes_batch, gt_classes_batch in train_dataloader:
            # forward pass
            loss = model(img_batch, gt_bboxes_batch, gt_classes_batch)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        loss_list.append(total_loss)

    return loss_list
# ------------- DEFINING DATASET CLASS --------------#


# ------------- DEFINING DATASET CLASS --------------#

img_width = 640
img_height = 480
image_dir = "./Selected_data"
str2label = {"pad": -1, "bee": 0}
label2str = {-1: "pad", 0: "bee"}
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL images to tensors
    # ... other transformations ...
])

train = VOCDataset(img_dir=os.path.join(image_dir,"train"), img_size=(img_height, img_width), str2label=str2label, label2str=label2str, transform=transform)
tr_dataloader = DataLoader(train, batch_size=2)

valid = VOCDataset(img_dir=os.path.join(image_dir,"valid"), img_size=(img_height, img_width), str2label=str2label, label2str=label2str, transform=transform)
val_dataloader = DataLoader(valid, batch_size=2)

# ------------- DEFINING MODEL CLASS --------------#

img_size = (img_height, img_width)
out_c = 2048
out_w = 20
out_h = 15
out_size = (out_h, out_w)
width_scale_factor = img_width // out_w
height_scale_factor = img_height // out_h

n_classes = len(str2label) - 1 # exclude pad idx
roi_size = (2, 2)

detector = TwoStageDetector(img_size, out_size, out_c, n_classes, roi_size)
detector.eval()

# ------------- TRAINING -----------#

learning_rate = 1e-3
n_epochs = 5

loss_list = training_loop(detector, learning_rate, tr_dataloader, n_epochs)
plt.plot(loss_list)
torch.save(detector.state_dict(), "model.pt")

detector.eval()
proposals_final, conf_scores_final, classes_final = detector.inference(val_dataloader, conf_thresh=0.99, nms_thresh=0.05)
# project proposals to the image space
prop_proj_1 = project_bboxes(proposals_final[0], width_scale_factor, height_scale_factor, mode='a2p')

# get classes
classes_pred_1 = [label2str[cls] for cls in classes_final[0].tolist()]
nrows, ncols = (1, 2)
fig, axes = plt.plot(figsize=(16, 8))
fig, _ = display_bbox(prop_proj_1, fig, classes=classes_pred_1)
plt.show()