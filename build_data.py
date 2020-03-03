import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


labels = pd.read_csv("train.csv")

class Cactus():
    IMG_SIZE = 32
    cactuscount = 0
    notcactuscount = 0
    # 0: not-cac    1: cac
    test_data_amount = [0, 0]       # trying to get 1000 cac and 1000 not-cac
    training_data = []
    test_data = []


    def make_dataset(self):
        for _, row in tqdm(labels.iterrows()):
            has_cac = row['has_cactus']
            try:
                path = os.path.join("train", row['id'])
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) / 255


                # first 1000 imgs of cac and not-cac goes to test dataset
                if(self.test_data_amount[has_cac] < 1000):
                    self.test_data_amount[has_cac] += 1
                    self.test_data.append([
                        np.array(img),
                        int(has_cac)])
                else:
                    self.training_data.append([
                        np.array(img),
                        int(has_cac)])
                
                if has_cac:
                    self.cactuscount += 1
                else:
                    self.notcactuscount += 1

            except Exception as e:
                print(str(e))

        print("Cactus / Total = ", 
        round(self.cactuscount / (self.cactuscount + self.notcactuscount), 4))

        print(len(self.training_data))


    # count cac
    def count(self, arr):
        sum = 0
        for _, is_cac in arr:
            sum += is_cac 
            # adding either 0 or 2
        return sum


    def save(self):
        np.random.shuffle(self.training_data)
        np.random.shuffle(self.test_data)

        np.save("training_data.npy", self.training_data)
        np.save("testing_data.npy", self.test_data)

        with open("info.txt", "a") as f:
            f.write(f"Time: {time.time()}\n")
            f.write(f"Whole dataset\n")
            f.write(f"Img size: {cac.IMG_SIZE}\n")
            f.write(f"Cactuses: {self.cactuscount}\n")
            f.write(f"Notcactuses: {self.notcactuscount}\n")
            f.write(f"Total: {self.cactuscount + self.notcactuscount}\n\n")


            f.write(f"Training dataset\n")

            f.write(f"Cactuses: {self.count(self.training_data)}\n")
            f.write(f"Training data samples: {len(self.training_data)}\n")
            f.write(f"Percentage: {self.count(self.training_data) / len(self.training_data)}\n")


            f.write(f"Testing dataset\n")
        
            f.write(f"Cactuses: {self.count(self.test_data)}\n")
            f.write(f"Test data samples: {len(self.test_data)}\n")
            f.write(f"Percentage: {self.count(self.test_data) / len(self.test_data)}\n")
            f.write("\n")




cac = Cactus()
cac.make_dataset()
cac.save()
