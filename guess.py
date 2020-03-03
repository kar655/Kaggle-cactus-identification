import torch
import os
import pandas as pd
import cv2
from tqdm import tqdm
from model_class import Model

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

# acc0.9805_20_lut_2020_18_09_38.pkl
# /home/karol/Desktop/kaggle/cactus/models/adam_20_lut_2020_18_08_02
model_name = "adam_20_lut_2020_18_08_02/acc0.9805_20_lut_2020_18_09_38.pkl"
path = os.path.join("models", model_name)

model = Model()
#model = model.load_state_dict(path)
model.load_state_dict(torch.load(path))
#model.eval()
#model.train()
#print(model)

input = torch.randn((32, 32)).view(-1, 1, 32, 32)
print(input)
out = model(input)
print(out)

with open("submission.csv", "a") as f:
    f.write("id,has_cactus\n")

    for img_name in tqdm(os.listdir("test")):

        path = os.path.join("test", img_name)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255
        img = torch.tensor(img).view(-1, 1, 32, 32).float()
        output = model(img)[0][1]
        f.write(f"{img_name},{output}\n")


