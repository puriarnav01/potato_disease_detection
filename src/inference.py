import torch
import cv2
import numpy as np
from torchvision import transforms as T
from network import CNN
from glob import glob
import matplotlib.pyplot as plt

aug = T.Compose([T.ToPILImage(), T.Resize((225,225)), T.ToTensor()])
model = CNN()
model.load_state_dict(torch.load("models/potato_cnn.pth"))

test_image_path = glob("data/test_images/*")

image_count = len(test_image_path)

class_map = {'Dry Rot': 0, 'Black Scurf': 1, 'Blackleg': 2, 'Common Scab': 3, 'Miscellaneous': 4, 'Healthy Potatoes': 5, 'Pink Rot': 6}

# for im_path in test_image_path:
#     image = cv2.imread(im_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = aug(image).unsqueeze(0)
#     gt = im_path.split("_")[-1].split(".")[0]
#     value = torch.argmax(model(image),axis=1).numpy()[0]
#     predicted = ""
#     for key,val in class_map.items():
#         if val == value:
#             predicted = key
    
#     print("GT:{}. Predicted:{}".format(gt, predicted))


fig,axs = plt.subplots(4,4,figsize=(15,10))

for i,ax in enumerate(axs.flatten()):
    rd_idx = np.random.randint(0,image_count)
    image = cv2.imread(test_image_path[rd_idx])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt = test_image_path[rd_idx].split("_")[-1].split(".")[0]
    image_pred = aug(image).unsqueeze(0)
    value = torch.argmax(model(image_pred),axis=1).numpy()[0]
    predicted = ""
    for key,val in class_map.items():
        if val == value:
            predicted = key

    ax.imshow(image)
    ax.axis("off")
    ax.set_title("GT: {} - Predicted: {}".format(gt,predicted))


plt.savefig("plots/inference.jpg")






# image = cv2.imread("data/test_images/test_image1_blackscurf.jpeg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = aug(image).unsqueeze(0)

# model.load_state_dict(torch.load("models/potato_cnn.pth"))

# value = model(image)
# print(torch.argmax(value,axis=1))