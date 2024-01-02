import pandas as pd
import glob as glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

file_list = glob.glob("data/*")

class_map = {}
counter = 0
for cls in file_list:
    cls_name = cls.split("/")[-1]
    class_map[cls_name] = counter
    counter+=1

print(file_list)
print(class_map)

df = pd.DataFrame()

for file in file_list:
    img_path = glob.glob(file+"/*jpg")
    #print(img_path)
    cls_label = file.split("/")[-1]
    cls_int = class_map[cls_label]
    #print(cls_int)
    for img in img_path:
        df = df.append({"image_path":img, "labels":cls_int, "class":cls_label},ignore_index=True)


num_samples = 10
new_output = [None] * num_samples

for i,sample in enumerate(range(num_samples)):
    new_output[i] = df.sample(n = len(df), replace = True)
    
data_new = pd.concat((new_output), ignore_index = True)

data_new.to_csv("data/dataset.csv")

def plot_samples(df):
    l = len(df)
    fig, axs = plt.subplots(3,4,figsize=(15,8))
    for i,ax in enumerate(axs.flatten()):
        rdm_idx = np.random.randint(0,l)
        image = cv2.cvtColor(cv2.imread(df.iloc[rdm_idx, 0]),cv2.COLOR_BGR2RGB)
        label = str(df.iloc[rdm_idx,2])
        ax.imshow(image)
        ax.set_title(label)
        ax.axis("off")
    fig.tight_layout()
    plt.savefig("plots/potato_images.jpg")

plot_samples(df)




