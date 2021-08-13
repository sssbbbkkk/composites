import sys
import os
import torch
import torch.nn as nn
from base_options import BaseOptions
import re
import MyNet
import math
import numpy as np
import cv2
import collections

def getTestSet(root):
    ImgWithLabel = collections.namedtuple('ImgWithLabel', ['img', 'input'])
    cnt = 0
    all_data = []
    pattern = re.compile(r'^\d+_img\.jpg$')
    for dirpath, dirnames, filenames in os.walk(root):
        if not 'image' in dirpath: continue
        for img_file_name in filenames:
            if not pattern.match(img_file_name): continue
            cnt += 1
            if cnt % 200 == 1: print(cnt)
            img_file_fullpath = os.path.join(dirpath, img_file_name)
            img = cv2.imread(img_file_fullpath)
            input_path = os.path.join(dirpath.replace('image', 'input'),
                                      img_file_name.replace('_img.jpg', '_input.txt'))

            if not os.path.exists(input_path): continue

            input_ref = np.array([230, 15, 0.4, 0.2, 4, 0.34, 0.6])
            # label_ref = np.array([139.6, 13.7, 5.6, 0.38, 900, 27, 200, 80])

            input = np.loadtxt(input_path) / input_ref


            if input.shape != (7,): continue
            all_data.append(ImgWithLabel(img=img, input=input))
    print('All data length: %d' % len(all_data))
    return all_data

def getResult(realData,symbol):
    input_ref = np.array([230, 15, 0.4, 0.2, 4, 0.34, 0.6])
    label_ref = np.array([139.6, 13.7, 5.6, 0.38, 900, 27, 200, 80])
    train_loss = 0.0
    final_results = []

    batch_cnt = 0

    for i, data in enumerate(realData):
        batch_cnt += 1
        img, input = data[0], data[1]
        img = img.transpose(-1, 0, 1)
        img = img.reshape(-1, *img.shape)
        img = torch.FloatTensor(img).to(device)
        input = torch.FloatTensor(input / input_ref).to(device)
        forward_result = symbol(img, input)
        forward_result = forward_result.detach().cpu().numpy()
        print (forward_result)
        final_results.append(forward_result * label_ref)
    final_result_mean = np.mean(final_results,axis=0)

    final_result_cov = np.std(final_results, axis=0, ddof=1)/final_result_mean
    #real_mean = np.array([139.4, 13.9, 5.1, 0.37, 700, 21, 154, 62])
    real_mean = np.array([139.6, 13.7, 5.6, 0.38, 900, 27, 200, 80])
    error = (final_result_mean-real_mean)/real_mean
    print(f'error is: {error*100}%' )
    print(f'cov is: {final_result_cov}')



if __name__=="__main__":
    model_name = 'MyNet'
    opt = BaseOptions().parse()

    device = torch.device("cuda:" + str(opt.gpu) if torch.cuda.is_available() else "cpu")

    # num_workers = cpu_count() if torch.cuda.is_available() else 0
    num_workers = 0
    model_name = 'MyNet'
    # symbol = NNet.get_zxj_resnet18()
    symbol = MyNet.get_symbol()
    # loss function L2 loss
    criterion_l2 = nn.MSELoss()

    # python inference.py --load_epoch latest
    if opt.load_epoch:
        mode_ends = '_latest.pth' if opt.load_epoch == 'latest' else '%05d.pth' % opt.load_epoch
        checkpoint_path = os.path.join(opt.checkpoints_dir, model_name + mode_ends)
        if os.path.exists(checkpoint_path):
            MyNet.load_model(symbol, checkpoint_path)
            print(checkpoint_path)

    symbol.to(device)
    symbol.eval()
    root = '../Data/testData'
    realData = getTestSet(root)
    getResult(realData, symbol)
