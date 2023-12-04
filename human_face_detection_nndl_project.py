#!/usr/bin/env python
# coding: utf-8

# In[1]:


# defining a few configurations

import torch

BATCH_SIZE = 4 # increase / decrease according to GPU memeory
RESIZE_TO = 512 # resize the image for training and transforms

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CLASSES = ['background', 'Face']
NUM_CLASSES = 2


# In[2]:


# check if cuda (GPU) is available

torch.cuda.is_available()


# In[3]:


# Split dataset into train, validation, and train sets

import pandas as pd
import shutil
import os

def train_valid_test_split(faces_csv=None, split=0.15):
    all_df = pd.read_csv(faces_csv)     
    
    # sample out 500 images
    all_df = all_df.sample(n=500, random_state=7)
    
    # Shuffle the CSV file rows.
    all_df.sample(frac=1)
    len_df = len(all_df)
    
    # Split into train/validation and test sets
    trainTest_split = int((1-split)*len_df)
    
    trainVal_df = all_df[:trainTest_split]
    test_df = all_df[trainTest_split:]
    
    # Further split train/validation set into train and validation sets
    lenTV_df = len(trainVal_df)
    
    trainVal_split = int((1-split)*lenTV_df)
    
    train_df = trainVal_df[:trainVal_split]
    valid_df = trainVal_df[trainVal_split:]
    
    return train_df, valid_df, test_df
    
train_df, valid_df, test_df = train_valid_test_split(faces_csv='Downloads/archive-2/faces.csv')


# In[4]:


import cv2
import numpy as np

# this class keeps track of the training and validation loss values...
# ... and helps to get the average for each epoch as well

class Averager:
    """""
    this class keeps track of the training and validation loss values...
    and helps to get the average for each epoch as well
    """""
    
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
        
def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


# In[5]:


import glob as glob
from torch.utils.data import Dataset, DataLoader

# Creating the dataset class

class Faces(Dataset):
    def __init__(self, dataset, width, height, dir_path="Downloads/archive-2/images"):
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.dataset = dataset
        
        # copy image names to list
        self.set_image_names = self.dataset['image_name'].tolist()
        
        # get all the image names in sorted order
        self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
        self.all_images = [image_path.split('/')[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)
        
        # cut down to only images present in dataset
        self.images = []
        
        for i in self.set_image_names:
            for j in self.all_images:
                if i == j:
                    self.images.append(i)
        
    def __getitem__(self, idx):
        
        # capture the image name and the full image path
        image_name = self.images[idx]
        image_path = os.path.join(self.dir_path, image_name)
        
        # read the image
        image = cv2.imread(image_path)
        
        # convert BGR to RGB color format and resize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        
        # channel first transposing
        image_resized = np.transpose(image_resized, (2, 0, 1))
             
        boxes = []
        labels = []

        # Copy bounding box coordinates and image dimensions
        filtered_df = self.dataset.loc[self.dataset['image_name'] == image_name]
        
        for i in range(len(filtered_df)):

            # xmax = left corner x-coordinates
            xmin = int(filtered_df['x0'].iloc[i])
            # xmax = right corner x-coordinates
            xmax = int(filtered_df['x1'].iloc[i])
            # ymin = left corner y-coordinates
            ymin = int(filtered_df['y0'].iloc[i])
            # ymax = right corner y-coordinates
            ymax = int(filtered_df['y1'].iloc[i])

            image_width = int(filtered_df['width'].iloc[i])
            image_height = int(filtered_df['height'].iloc[i])

            # resize the bounding boxes according to the...
            # ... desired `width`, `height`
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            yamx_final = (ymax/image_height)*self.height

            boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])
            labels.append(1) # 1 because there is only one class
        
        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # crowd instances
        if boxes.shape[0] > 1:
            iscrowd = torch.ones((boxes.shape[0],), dtype=torch.int64)
        else:
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            
        # label to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        return image_resized, target
    
    def __len__(self):
        return len(self.set_image_names)


# In[6]:


train_dataset = Faces(train_df, RESIZE_TO, RESIZE_TO)
valid_dataset = Faces(valid_df, RESIZE_TO, RESIZE_TO)

# defining train and validation sets data loaders

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}\n")


# In[7]:


import matplotlib.pyplot as plt

# function to visualize sample

def visualize_sample(image, target):
    for box in target['boxes']:
        cv2.rectangle(
            image, 
            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            (0, 255, 0), 1
        )
    
    plt.imshow(image)
    plt.axis('off')
    plt.show()

NUM_SAMPLES_TO_VISUALIZE = 5
for i in range(NUM_SAMPLES_TO_VISUALIZE):
    image, target = train_dataset[i]
    image = np.transpose(image, (1, 2, 0))
    visualize_sample(image, target)


# In[8]:


# defining model

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
def create_model(num_classes):
    
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model


# In[9]:


# function for running training iterations

def train(train_data_loader, model):
    print('Training')
    global train_itr
    global train_loss_list
    
     # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        
        images = list(torch.from_numpy(image).to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        losses.backward()
        optimizer.step()
        train_itr += 1
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list


# In[10]:


# function for running validation iterations

def validate(valid_data_loader, model):
    print('Validating')
    global val_itr
    global val_loss_list
    
    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(torch.from_numpy(image).to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list


# In[11]:


# training the model

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time

plt.style.use('ggplot')

NUM_EPOCHS = 10 # number of epochs to train for

# initialize the model and move to the computation device
model = create_model(num_classes=NUM_CLASSES)
model = model.to(DEVICE)
# get the model parameters
params = [p for p in model.parameters() if p.requires_grad]
# define the optimizer
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
# initialize the Averager class
train_loss_hist = Averager()
val_loss_hist = Averager()
train_itr = 1
val_itr = 1
# train and validation loss lists to store loss values of all...
# ... iterations till ena and plot graphs for all iterations
train_loss_list = []
val_loss_list = []
# name to save the trained model with
MODEL_NAME = 'model'

# start the training epochs
for epoch in range(NUM_EPOCHS):
    print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
    
    # reset the training and validation loss histories for the current epoch
    train_loss_hist.reset()
    val_loss_hist.reset()
    
    # start timer and carry out training and validation
    start = time.time()
    train_loss = train(train_loader, model)
    val_loss = validate(valid_loader, model)
    
    print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
    print(f"Epoch #{epoch+1} validation loss: {val_loss_hist.value:.3f}")   
    end = time.time()
    print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch+1}")

    if (epoch+1) == NUM_EPOCHS: # save loss plots and model once at the end
        # create two subplots, one for each, training and validation
        figure_1, train_ax = plt.subplots()
        figure_2, valid_ax = plt.subplots()        
        train_ax.plot(train_loss, color='blue')
        train_ax.set_xlabel('iterations')
        train_ax.set_ylabel('train loss')
        valid_ax.plot(val_loss, color='red')
        valid_ax.set_xlabel('iterations')
        valid_ax.set_ylabel('validation loss')
        figure_1.savefig(f"Downloads/archive-2/train_loss_{epoch+1}.png")
        figure_2.savefig(f"Downloads/archive-2/valid_loss_{epoch+1}.png")
        torch.save(model.state_dict(), f"Downloads/archive-2/model{epoch+1}.pth")

train_ax.plot(train_loss, color='blue')
train_ax.set_xlabel('iterations')
train_ax.set_ylabel('train loss')

valid_ax.plot(val_loss, color='red')
valid_ax.set_xlabel('iterations')
valid_ax.set_ylabel('validation loss')
plt.show()


# In[18]:


import random
import glob
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

model.eval()

# define the detection threshold...
# ... any detection having a score below this will be discarded
detection_threshold = 0.8

# copy image names to list
stest_image_names = test_df['image_name'].tolist()

# get all the image names in sorted order
dir_path = "Downloads/archive-2/images"
image_paths = glob.glob(f"{dir_path}/*.jpg")
all_images = [image_path.split('/')[-1] for image_path in image_paths]
all_images = sorted(all_images)

# get paths of only images present in the dataset
test_images = []

for i in stest_image_names:
    for j in all_images:
        if i == j:
            test_images.append(os.path.join(dir_path, i))

print(f"Test instances: {len(test_images)}")

for i in random.sample(range(len(test_images)), 5):
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to the front
    image = np.transpose(image, (2, 0, 1)).astype(float)
    # convert to tensor
    # Assuming device is either 'cuda' if available, otherwise 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move tensor to the selected device
    image = torch.tensor(image, dtype=torch.float).to(device)
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        outputs = model(image)

    # load all detections to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()

        # draw the bounding boxes
        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0, 0, 255), 2)
        image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()
    print(f"Image {i + 1} done...")
    print('-' * 50)

print('TEST PREDICTIONS COMPLETE')


# In[ ]:




