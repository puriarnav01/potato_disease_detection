import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms as T
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split

class PotatoDataset(Dataset):
    def __init__(self,path,transform):
        self.data = pd.read_csv(path)
        del self.data["Unnamed: 0"]
        self.image_paths = self.data["image_path"]
        self.image_labels = self.data["labels"]
        self.transform = transform
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        label = self.image_labels[idx]
        label = torch.tensor(label).long()
        return image, label
    
    def __len__(self):
        return len(self.image_labels)

#train_data = PotatoDataset("data/dataset.csv", aug)
#

data = pd.read_csv("data/dataset.csv")
train, test = train_test_split(data, test_size=.1)
del train["Unnamed: 0"]
del test["Unnamed: 0"]
train = train.reset_index()
test = test.reset_index()
train.to_csv("data/train.csv")
test.to_csv("data/test.csv")

train_aug = T.Compose([T.ToPILImage(), T.Resize((225,225)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(), 
                       T.RandomRotation(90),T.ToTensor()])
test_aug = T.Compose([T.ToPILImage(), T.Resize((225,225)), T.ToTensor()])

train_data = PotatoDataset("data/train.csv", train_aug)
train_loader = DataLoader(train_data, shuffle=True, drop_last=True,batch_size=32)

test_data = PotatoDataset("data/test.csv", test_aug)
test_loader = DataLoader(test_data, shuffle=False, batch_size=len(test_data))

# for X,y in test_loader:
#     print(X.shape)
#     print(y.shape)