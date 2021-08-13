import os
import sys
import random
import re
import torch
import numpy as np
import collections
import torch.utils.data as data
import cv2
from torchvision import transforms

ImgWithLabel = collections.namedtuple('ImgWithLabel', ['img','input', 'label'])


class FaceDataSet(data.Dataset):
    def __init__(self, root, train_number=None):
        super(FaceDataSet, self).__init__()
        self.all_data = []
        cnt = 0
        pattern = re.compile(r'^\d+_img\.jpg$')
        for dirpath, dirnames, filenames in os.walk(root): #读图读输入
            if not 'image' in dirpath: continue
            for img_file_name in filenames:
                if not pattern.match(img_file_name): continue
                cnt += 1
                if cnt % 200 == 1: print(cnt)
                img_file_fullpath = os.path.join(dirpath, img_file_name)
                img = cv2.imread(img_file_fullpath)
                input_path = os.path.join(dirpath.replace('image','input'),img_file_name.replace('_img.jpg','_input.txt'))
                label_file_fullpath = os.path.join(dirpath.replace('image','target'),img_file_name.replace('_img.jpg','_target.txt'))
                if not os.path.exists(label_file_fullpath) or not os.path.exists(input_path): continue


                input_ref = np.array([230,15,0.4,0.2,4,0.34,0.6])
                label_ref = np.array([139.6,13.7,5.6,0.38,900,27,200,80]) #输入归一化

                input = np.loadtxt(input_path) / input_ref
                label = np.loadtxt(label_file_fullpath) / label_ref

                if input.shape != (7,): continue
                if label.shape != (8,): continue
                self.all_data.append(ImgWithLabel(img=img, input=input,label=label))
        if train_number is not None: self.all_data = self.all_data[:train_number]
        print('All data length: %d' %  len(self.all_data) )
        sys.stdout.flush()

    def __getitem__(self, index):
        img = self.all_data[index].img
        input = self.all_data[index].input
        label = self.all_data[index].label
        img = img.transpose(-1, 0, 1)
        img=torch.FloatTensor(img)
        input= torch.FloatTensor(input)
        label= torch.FloatTensor(label)
        return {'img': img,'input': input, 'label': label}

    def __len__(self):
        return len(self.all_data)


if __name__=='__main__':
    trainset = FaceDataSet('../data/testData')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers = 0, drop_last = True)


    for i, batch in enumerate(trainloader):
        print(batch['img'].size())
        print(batch['input'].size())
        print(batch['label'].size())
        sys.exit()
