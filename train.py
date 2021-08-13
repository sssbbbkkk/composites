import sys
import os
import torch.optim as optim
import torch
import torch.nn as nn
from base_options import BaseOptions
from face_dataset import FaceDataSet
import MyNet
import cv2
import numpy as np
import math

model_name = 'MyNet'


if __name__=="__main__":



    opt = BaseOptions().parse()

    device = torch.device("cuda:" + str(opt.gpu) if torch.cuda.is_available() else "cpu")

    # num_workers = cpu_count() if torch.cuda.is_available() else 0
    num_workers = 0
    model_name = 'MyNet'
    # symbol = NNet.get_zxj_resnet18()
    symbol = MyNet.get_symbol()
    # loss function L2 loss
    criterion_l2 = nn.MSELoss()

    if opt.load_epoch:
            mode_ends = '_latest.pth' if opt.load_epoch == 'latest' else '%05d.pth'%opt.load_epoch
            checkpoint_path = os.path.join(opt.checkpoints_dir, model_name + mode_ends)
            if os.path.exists(checkpoint_path): MyNet.load_model(symbol, checkpoint_path)

    symbol.to(device)
    symbol_optimizer = optim.Adam(symbol.parameters(), lr=opt.lr)

    trainset = FaceDataSet('../data/trainData')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
                                              num_workers=num_workers, drop_last=True)

    testset = FaceDataSet('../data/testData')
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=True, num_workers=num_workers,
                                             drop_last=True)

    for epoch in range(10000):
        train_loss = 0.0
        if opt.train_mode == 'train':
            symbol.train()
        else:
            symbol.eval()
        #get training samples
        for j in range(1):
            try:
                batch_y = next(train_iter)
            except:
                train_iter = iter(trainloader)
                batch_y = next(train_iter)
            try:
                batch_y_test = next(train_testiter)
            except:
                train_testiter = iter(testloader)
                batch_y_test = next(train_testiter)

            # forward
            y_imgs, y_inputs, y_labels = batch_y['img'], batch_y['input'], batch_y['label']
            y_imgs,y_inputs, y_labels = y_imgs.to(device),y_inputs.to(device), y_labels.to(device)
            y_out = symbol(y_imgs,y_inputs)
            # loss function L2 loss
            loss = criterion_l2(y_out, y_labels)


            # backward
            symbol_optimizer.zero_grad()
            loss.backward()
            symbol_optimizer.step()
            train_loss += loss.item()
            #print training result
            result_info = '[epoch:%d, iter:%5d]' % (epoch + 1, j + 1)
            result_info += model_name + ':%.5f ' % (train_loss / 100.0)
            print(result_info)
            train_loss = 0.0
            sys.stdout.flush()

            # save the model
            if epoch % 20 == 1:
                torch.save(symbol.state_dict(),
                           os.path.join(opt.checkpoints_dir, model_name + '%05d.pth' % (epoch + 1)))
                torch.save(symbol.state_dict(), os.path.join(opt.checkpoints_dir, model_name + '_latest.pth'))

            # Test the result

            if epoch % 20 == 1:
                train_loss = 0.0
                symbol.eval()
                batch_cnt = 0

                # for i, data in enumerate(trainloader):
                for i, data in enumerate(testloader):
                    batch_cnt += 1
                    imgs, inputs, labels = data['img'],data['input'], data['label']
                    imgs,inputs, labels = imgs.to(device),inputs.to(device), labels.to(device)
                    outputs = symbol(imgs,inputs)
                    loss = criterion_l2(outputs, labels)
                    train_loss += loss.item()

                result_info = 'Test Result:'
                result_info += model_name + ':%.5f ' % (math.sqrt(train_loss / batch_cnt))
                print(result_info)
                sys.stdout.flush()

            #Test the forward result
            if epoch % 100 == 1:
                symbol.eval();
                input_ref = np.array([230, 15, 0.4, 0.2, 4, 0.34, 0.6])
                label_ref = np.array([139.6, 13.7, 5.6, 0.38, 900, 27, 200, 80])

                real_img = cv2.imread('../data/realData/image/00001_img.jpg')
                real_input = np.loadtxt('../data/realData/input/00001_input.txt')
                real_target = np.loadtxt('../data/realData/target/00001_target.txt')
                real_img = real_img.transpose(-1, 0, 1)
                real_img = real_img.reshape(-1, *real_img.shape)
                real_img = torch.FloatTensor(real_img).to(device)

                real_input = torch.FloatTensor(real_input/input_ref).to(device)
                forward_result = symbol(real_img,real_input)
                forward_result = forward_result.detach().cpu().numpy()
                final_result = forward_result*label_ref
                error = (final_result-real_target)/real_target*100
                result_info =f'Forward Result :{final_result} Real Target: {real_target} Average error:{error}%'

                print(result_info)
                sys.stdout.flush()


