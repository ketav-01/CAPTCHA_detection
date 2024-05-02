import torch
import torchvision
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from model import SpinalVGG


num_epochs = 50
learning_rate = 0.005
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CapDataset(Dataset):
    def __init__(self, annotation_file):
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transforms.Compose([transforms.RandomRotation(30)])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_vec = torch.tensor(self.annotations.iloc[index, 1:], dtype=torch.float)
        img = torch.reshape(img_vec,(1,28,28))
        # img = torch.transpose(img,1,2)
        
        y_label = torch.tensor(self.annotations.iloc[index, 0],dtype=torch.long)
        if (y_label <= 61 ):
            img = torch.transpose(img,1,2)

        return (img, y_label)
    
df = pd.read_csv("DATA\\combined-dataset.csv")
num_samples = df.shape[0]

dataset = CapDataset("DATA\\combined-dataset.csv")
train_set, validation_set = torch.utils.data.random_split(dataset,[((int)(0.9*num_samples)),num_samples - ((int)(0.9*num_samples))])
train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(dataset=validation_set, shuffle=True, batch_size=batch_size)

examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])

examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr1 = learning_rate

model1 = SpinalVGG().to(device)
# model1.load_state_dict(torch.load("WEIGHT\\model.h5"))

criterion = nn.CrossEntropyLoss()
optimizer1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)

total_step = len(train_loader)
best_accuracy1 = 0
for epoch in tqdm(range(num_epochs)):
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model1(images)
        loss1 = criterion(outputs, labels)

        # Backward and optimize
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        
        if i % 1000 == 0:
            print ("Ordinary Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss1.item()))

    model1.eval()
    with torch.no_grad():
        correct1 = 0
        total1 = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            
            outputs = model1(images)
            _, predicted = torch.max(outputs.data, 1)
            total1 += labels.size(0)
            correct1 += (predicted == labels).sum().item()
        
        if best_accuracy1>= correct1 / total1:
            curr_lr1 = learning_rate*np.isscalar(pow(np.random.rand(1),3))
            update_lr(optimizer1, curr_lr1)
            print('Test Accuracy of NN: {} % Best: {} %'.format(100 * correct1 / total1, 100*best_accuracy1))
        else:
            best_accuracy1 = correct1 / total1
            net_opt1 = model1
            print('Test Accuracy of NN: {} % (improvement)'.format(100 * correct1 / total1))
        
        torch.save(model1.state_dict(), "WEIGHT//model.h5")
            
        model1.train()