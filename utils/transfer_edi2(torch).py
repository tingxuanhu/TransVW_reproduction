import os

import numpy as np
import skimage.transform
from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.ynet3d import *
from config import *

torch.manual_seed(1)
# transform = transforms.Compose([
#     transforms.ToTensor(),  # 转为tensor且归一化 （此前已经归一化）
# ])


# imitate data_loader.py file to establish our own dataset for pneumonia classification
class MyData(Dataset):
    def __init__(self, root, train=True, test=False):
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-3]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-3]))

        imgs_num = len(imgs)
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]  # 将train中数据的70%给train
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]  # 剩下的30%做验证集

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        input_rows = 128
        input_cols = 128
        input_deps = 64

        img_path = self.imgs[index]
        data = np.load(img_path)
        data = skimage.transform.resize(data, (input_rows, input_cols, input_deps), preserve_range=True)
        data = np.expand_dims(data, axis=0)

        if self.test:
            label = int(self.imgs[index].split('.')[-3])
        else:
            label = 1 if '.1.' in img_path else 0

        return data, label


# customize dataset
root_dir = '/home/data/tingxuan/demo/classification'
train_dataset = MyData(root_dir, train=True)
val_dataset = MyData(root_dir, train=False)

# using dataloader
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)


# prepare the 3D model
class TargetNet(nn.Module):
    def __init__(self, base_model, n_class=1):
        super(TargetNet, self).__init__()
        self.base_model = base_model

        self.dense_1 = nn.Linear(in_features=512, out_features=128, bias=True)
        self.dense_2 = nn.Linear(in_features=128, out_features=n_class, bias=True)

        # self.dense_1 = nn.Linear(in_features=512, out_features=n_class, bias=True)

    def forward(self, x):
        self.base_model(x)
        self.base_out = self.base_model.out512
        # glb_avg_pool is for (N--batch_size,C--channels,H,W), not for (N,H,W,C)
        self.out_glb_avg_pool = F.avg_pool3d(input=self.base_out,
                                             kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0], -1)
        self.linear_out = self.dense_1(self.out_glb_avg_pool)
        self.linear_out2 = self.dense_2(self.linear_out)
        final_out = F.relu(self.linear_out2)

        # final_out = F.relu(self.linear_out)
        return final_out


base_model = UNet3D()


# ----------------------- Load pre-trained weights -----------------------------
weight_dir = "Checkpoints/en_de/TransVW_chest_ct.pt"
checkpoint = torch.load(weight_dir)

state_dict = checkpoint['state_dict']
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

delete = [key for key in state_dict if "projection_head" in key]
for key in delete:
    del state_dict[key]
delete = [key for key in state_dict if "prototypes" in key]
for key in delete:
    del state_dict[key]

for key in state_dict.keys():
    if key in base_model.state_dict().keys():
        base_model.state_dict()[key].copy_(state_dict[key])
        # print("Copying {} <---- {}".format(key, key))
    elif key.replace("classficationNet.", "") in base_model.state_dict().keys():
        base_model.state_dict()[key.replace("classficationNet.", "")].copy_(state_dict[key])
        # print("Copying {} <---- {}".format(key.replace("classficationNet.", ""), key))
    else:
        print("Key {} is not found".format(key))

target_model = TargetNet(base_model)
device = torch.device("cuda")
target_model.to(device)

target_model = nn.DataParallel(target_model, device_ids=[i for i in range(torch.cuda.device_count())])

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(target_model.parameters(), lr=.0001, momentum=.9, weight_decay=0.0, nesterov=False)


# ------------------------------------------- train the model --------------------------------------------
# for epoch in range(initial_epoch, config.nb_epoch):   # 改成迭代轮数
for epoch in range(2):   # 改成迭代轮数
    target_model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.float().to(device), y.float().to(device)
        # print(x.shape)

        # x, y = x.float(), y.float()

        pred = torch.sigmoid(target_model(x)).view(y.size())
        print(pred)
        print(y)
        loss = criterion(pred, y)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# https://blog.csdn.net/robot_learner/article/details/122169847  --> 训练过程  测试结果书写

