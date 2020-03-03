import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import os
import cv2

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

from model_class import Model
# MODEL DEF 

# model = Model()
# input = torch.randn((32, 32)).view(-1, 1, 32, 32)
# print(input)
# out = model(input)
# print(out)
# print(torch.argmax(out))

model = Model().to(device)

lr = 0.001
# not-cac / cac
# 3364 / 12136
not_cac = 3364
cac = 12136
#weights = torch.tensor((15500 - 12136, 12136)) / 15500.
#weights = torch.tensor((12136, 15500 - 12136)) / 15500.     # czemu to

# chyba tak powinno byc
weights = torch.tensor((cac / not_cac, cac / cac))

weights = weights.double().to(device)
print(weights)


optimizer = optim.Adam(model.parameters(), lr=lr)           # czemu to
#optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
loss_function = nn.CrossEntropyLoss(weight=weights)         # czemu to
#loss_function = nn.CrossEntropyLoss()

EPOCHS = 50
BATCH_SIZE = 100
IMG_SIZE = 32

training_data = np.load("training_data.npy", allow_pickle=True)
testing_data = np.load("testing_data.npy", allow_pickle=True)

train_x = torch.tensor([i[0] for i in training_data])
train_y = torch.tensor([i[1] for i in training_data])

test_x = torch.tensor([i[0] for i in testing_data])
test_y = torch.tensor([i[1] for i in testing_data])

print(len(test_x))

plt.imshow(test_x[0].float().numpy(), cmap="gray")
plt.show()
print(train_x.size())
print(train_x[0].size())
print(train_x[0].dtype)
print(test_x[0])
print(test_y[0])


def calculate_accuracy(data_x, data_y):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(data_x), BATCH_SIZE):
            batch_x = data_x[i:i + BATCH_SIZE].view(-1, 1, IMG_SIZE, IMG_SIZE).float().to(device)
            batch_y = data_y[i:i + BATCH_SIZE].long().to(device)
            outputs = torch.argmax(model(batch_x).double(), dim=1)
            total += BATCH_SIZE
            correct += (batch_y == outputs).sum().item()

    return round(correct / total, 4)
    
file_name = "adam{}".format(time.strftime("_%d_%b_%Y_%H_%M_%S"))
path = os.path.join(os.getcwd(), "models", file_name)
os.mkdir(path)
#path = os.path.join(path , file_name)
#os.system(f"mkdir {file_name}")

with open(os.path.join(path, "model.log"), "a") as f:
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_x), BATCH_SIZE)):
            batch_x = train_x[i:i + BATCH_SIZE].view(-1, 1, IMG_SIZE, IMG_SIZE).float().to(device)
            batch_y = train_y[i:i + BATCH_SIZE].long().to(device)

            model.zero_grad()
            outputs = model(batch_x).double()
            loss = loss_function(outputs, batch_y)

            loss.backward()
            optimizer.step()

            size = BATCH_SIZE
            random_start = np.random.randint(len(test_x) - size)
            f.write(f"{round(time.time(), 3)},{calculate_accuracy(test_x[random_start:random_start + size], test_y[random_start:random_start + size])},{loss.item()}\n")

        #time.strftime("%d %b %Y %H:%M:%S")
        print(f"Epoch: {epoch + 1} / {EPOCHS}\tLoss: {loss.item()}")
        acc = calculate_accuracy(test_x, test_y)
        print(f"Accuracy training data: ", calculate_accuracy(train_x, train_y))
        print(f"Accuracy testing data: ", acc)
        
        torch.save(model.state_dict(), os.path.join(path, "acc{}{}.pkl".format(acc, time.strftime("_%d_%b_%Y_%H_%M_%S"))))
        

