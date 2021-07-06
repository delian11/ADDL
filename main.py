import numpy as np
import torch
import torch.nn as nn

from torchvision import transforms
from torch.nn import functional as F
from utils.losses import *
from utils.metrics import *
from utils.plot import *
from utils.save_model import *
from utils.set_path import *
from datasets.load_data import load_data

import os
import math
import argparse
import shutil
from numpy import shape, where, in1d

from datetime import datetime

from PIL import Image

import socket
hostname = socket.gethostname()


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0', help='comma separated list of GPU(s) to use.')
parser.add_argument('--load', help='load model')
parser.add_argument('--weight', default='inter', type=str, help='multi-task weighting: manual, inter')
parser.add_argument('--dataset', help='dataset', type=str, default='RAF')
parser.add_argument('--lr', help='base learning rate', type=np.float32, default=0.0001)
parser.add_argument('--d_train_repeat',type=int, default=1, help='the train interval of discriminator')
parser.add_argument('--miu', help='the weight of fit loss', type=np.float32, default=0.1)
parser.add_argument('--dim', help='the dim of feature', type=int, default=128, choices=[256, 128, 64])
parser.add_argument('--plot', action='store_true', help="whether to plot features for every epoch")
parser.add_argument('--pretrain', action='store_true', help="whether pre-train")
parser.add_argument('--fold', default=1, type=int, help='cross validation fold, for CK OULU MMI')
parser.add_argument('--inter', default=1, type=int, help='inter of D')
parser.add_argument('--bs', type=int, default=16, help='batch size')
parser.add_argument('--nepochs', type=int, default=40, help='total epochs')
parser.add_argument('--resume', action='store_true', help="whether loading the trained models")
parser.add_argument('--resume_epoch', type=int, default=0, help='total epochs')
args = parser.parse_args()
print(args)


def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
        lr +=[ param_group['lr'] ]
    return lr

def get_set_gpus(gpu_ids):
    # get gpu ids
    if len(gpu_ids) > 1: 
        str_ids = gpu_ids.split(',')
        gpus = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                gpus.append(id)
    else:
        gpus = [int(gpu_ids)]
    # set gpu ids
    if len(gpus) > 0:
        torch.cuda.set_device(gpus[0])
    return gpus

def train(net, model1, model2, D, train_data, criterion, num_classes, writer, lambda_weight, epoch):
    # train model
    net = net.train()
    lr = get_learning_rate(optimizer)
    index = epoch

    print("\n==> Epoch %d/%d, lr %g. cls %.2f, D %.2f." \
    %(epoch, num_epochs, lr[0], lambda_weight[0,index], lambda_weight[1,index]))

    train_correct = 0
    train_total = 0
    cost = [0] * 4

    global plot
    plot = False
    if (index+1) % 2 == 0 and args.plot:
        plot = True
        all_features, all_labels = [], []

    for batch_idx, (im, label) in enumerate(train_data):
        if torch.cuda.is_available():
            im = im.cuda()
            label = label.cuda()
        
        # extract disturbing features
        feat, masks_1 = model1(im)
        feat_real_1 = feat.detach()
        
        feat, masks_2 = model2(im)
        feat_real_2 = feat.detach()
        
        out_feat, out_logit, feat_dist, masks  = net(im)
        cls_loss = criterion(out_logit, label)

        # Using D to surpervise S_d
        BCE = torch.nn.BCEWithLogitsLoss()
        if batch_idx % args.d_train_repeat == 0:
            # Compute loss with predict feat
            out_d = D(feat_dist)
            d_loss_pre = BCE(out_d, torch.zeros(out_d.size()).cuda())
            
            # Compute loss with groundtruth feat
            out_d = D(feat_real_1)
            d_loss_gt_1 = BCE(out_d, torch.ones(out_d.size()).cuda())
            out_d = D(feat_real_2)
            d_loss_gt_2 = BCE(out_d, torch.ones(out_d.size()).cuda())
            
            # Backward and optimize.
            d_loss = (d_loss_pre + d_loss_gt_1 + d_loss_gt_2) #* lambda_gan
            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()
        
        out_d = D(feat_dist)
        dist_loss = BCE(out_d, torch.ones(out_d.size()).cuda())

        mask_loss_1 = mask_loss(masks, masks_1)
        mask_loss_2 = mask_loss(masks, masks_2)
        mask_losses = 0.05 * (mask_loss_1 + mask_loss_2)

        ddm_loss = lambda_weight[0, index] * cls_loss + lambda_weight[1, index] * dist_loss + mask_losses

        _, pred_label = torch.max(out_logit.data, dim=1)
        train_total += label.size(0)
        train_correct += (pred_label == label.data).cpu().sum()
        
        # backward
        optimizer.zero_grad()
        d_optimizer.zero_grad()
        ddm_loss.backward(retain_graph=True)
        optimizer.step()
        

        if plot:
            if torch.cuda.is_available():
                all_features.append(out_feat.data.cpu().numpy())
                all_labels.append(label.data.cpu().numpy())
            else:
                all_features.append(out_feat.data.numpy())
                all_labels.append(label.data.numpy())

        cost[0] += d_loss.data.cpu().numpy()
        cost[1] += cls_loss.data.cpu().numpy()
        cost[2] += dist_loss.data.cpu().numpy()
        cost[3] += mask_losses.data.cpu().numpy()


        if (batch_idx+1) % 50 == 0:
            print('Batch {:4d}/{}. E: Cls loss: {:.6f}, d loss: {:.6f}, mask loss:  {:.6f}, Acc (%): {:.4f}; D: d loss: {:.6f}'.format
                (batch_idx+1, len(train_data), cls_loss.data.cpu().numpy(), dist_loss.data.cpu().numpy(), mask_losses.data.cpu().numpy(), float(train_correct) / train_total * 100, d_loss.data.cpu().numpy())
                )

    print("Train--> E: cls loss: {:.6f}, d loss: {:.6f}, D: d_loss: {:.6f} | Accuracy (%): {:.4f}.\n".format
    (cost[1] / len(train_data), cost[2] / len(train_data), cost[0] / len(train_data), float(train_correct) / train_total * 100))

    # saved to tensorboardX
    writer.add_scalar('Train adv loss', cost[0] / len(train_data), epoch)
    writer.add_scalar('Train cls loss', cost[1] / len(train_data), epoch)
    writer.add_scalar('Train dist loss', cost[2] / len(train_data), epoch)
    writer.add_scalar('Train mask loss', cost[3] / len(train_data), epoch)
    writer.add_scalar('Train Acc', float(train_correct) / train_total * 100, epoch)

    if plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(all_features, all_labels, num_classes, epoch, plot_dir, args.dataset, prefix='train'+str(index))



#test model
def test(net, D, test_data, criterion, num_classes, writer, best_metric, best_epoch, best_mat, lambda_weight, index):
    if test_data is not None:
        net = net.eval()
        test_cost = 0
        test_acc = 0
        total = 0
        correct = 0
        true_label = []
        pre_label = []

        if plot:
            all_features, all_labels = [], []
        with torch.no_grad():
            for im, label in test_data:
                if torch.cuda.is_available():
                    im = im.cuda()
                    label = label.cuda()
                
                out_feat, out_logit, feat_dist, masks = net(im)
                loss = criterion(out_logit, label)

                if plot:
                    if torch.cuda.is_available():
                        all_features.append(out_feat.data.cpu().numpy())
                        all_labels.append(label.data.cpu().numpy())
                    else:
                        all_features.append(out_feat.data.numpy())
                        all_labels.append(label.data.numpy())

                _, pred_label = torch.max(out_logit.data, dim=1)
                total += label.size(0)
                correct += (pred_label == label.data).cpu().sum()
                test_cost += loss.data.cpu().numpy()

                true_label.append(label.data.cpu().numpy())
                pre_label.append(pred_label.cpu().numpy())

        test_acc = float(correct) / total

        # saved to tensorboardX
        writer.add_scalar('Test cls loss', test_cost / len(test_data), epoch)
        writer.add_scalar('Test Acc', test_acc, epoch)
        

        if plot:
            all_features = np.concatenate(all_features, 0)
            all_labels = np.concatenate(all_labels, 0)
            plot_features(all_features, all_labels, num_classes, epoch, plot_dir, args.dataset, prefix='test'+str(index))

        true_label = np.concatenate(true_label)
        pre_label = np.concatenate(pre_label)

        acc_mat, _ = Get_confusion_matrix(true_label, pre_label)
        if test_acc > best_metric:
            best_metric = test_acc
            best_mat = acc_mat
            best_epoch = index
            # save_model(save_dir, net, '_best', optimizer)
            save_best_model(args.dataset, best_metric, best_epoch, save_dir, net)
        print("Test --> cls loss: {:.6f}, Acc (%): {:.4f}, Best Acc : {:.4f} (epoch {}).".format(test_cost / len(test_data), test_acc * 100, best_metric * 100, best_epoch), end='')

        return best_metric, best_epoch, best_mat
    

if __name__ == "__main__":
    # setting parameters
    if args.gpu:
        gpus = get_set_gpus(args.gpu)
        print('==> Currently use GPU: {}'.format(gpus))

    if hostname == 'ubuntu-server':
        root = '/media/data3/data/ruan'
    elif hostname == 'yanyan3':
        root = '/media/data/ruan_data'
    elif hostname == 'ubuntu-ruan':
        root = '/media/data'

    data_root = root
    save_root = root

    # set model_save_path, writer_path, plot_path
    if args.plot:
        save_dir, writer, plot_dir = set_path(save_root)
    else:
        save_dir, writer = set_path(save_root)

    # load data
    train_loader, test_loader, num_classes = load_data(data_root, args.dataset, args.bs, args.fold)

    # model
    from models.DDM import DDM
    from models.Discriminator import Discriminator
    num_blocks = [2, 2, 2, 2]
    filters = [64,64,128,256,512]
    dim = args.dim

    net = DDM(num_blocks, filters, num_classes, dim)
    D = Discriminator(in_dim=dim, d=64)
    print('Network established! Parameters:', sum(param.numel() for param in net.parameters()))

    if args.pretrain:
        model_dict = net.state_dict()
        print('--Finetuning on AffectNet pretrained model..')
        read_path = root + '/Model/DDL/affectnet/Dwa_lr0.01_K3.0_T2.0/epoch30.t7'
        pretrained_dict = torch.load(read_path, map_location=lambda storage, loc: storage.cuda(gpus[0]))['state_dict']
        # for k, v in pretrained_dict.items():
        #     if k in model_dict:
        #         print(k)
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    # load pretrained model
    from models.DFEM import DFEM
    model1 = DFEM(num_blocks, filters, num_classes, dim)
    model_dict = model1.state_dict()
    print('Using Multi-PIE trained model to extract disturbing (pose, illumination, subject) features..')
    read_path = root + '/Model/disturbance/PIE/sub_exp_pose_illu/epoch20.t7'
    pretrained_dict = torch.load(read_path, map_location=lambda storage, loc: storage.cuda(gpus[0]))['state_dict']
    pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model1.load_state_dict(model_dict)

    model2 = DFEM(num_blocks, filters, num_classes, dim)
    model_dict = model2.state_dict()
    print('Using RAF-DB trained model to extract disturbing (gender, race, age) features..')
    read_path = root + '/Model/disturbance/RAF/exp_gen_race_age/epoch20.t7'
    pretrained_dict = torch.load(read_path, map_location=lambda storage, loc: storage.cuda(gpus[0]))['state_dict']
    pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model2.load_state_dict(model_dict)

    # multi-gpu set
    if torch.cuda.is_available():
        if len(gpus) > 1:
            net = torch.nn.DataParallel(net, device_ids=gpus)
        net = net.cuda(gpus[0])
        model1 = model1.cuda(gpus[0])
        model2 = model2.cuda(gpus[0])
        D = D.cuda(gpus[0])
    print('')

    d_optimizer = torch.optim.Adam(D.parameters(), args.lr, [0.5, 0.999], weight_decay=5e-4)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=[0.5, 0.999], weight_decay=5e-4)
    # d_optimizer = torch.optim.SGD(D.parameters(), args.lr, weight_decay=5e-4, nesterov=True, momentum=0.9)
    # optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=5e-4, nesterov=True, momentum=0.9)
    if args.pretrain:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,18,25,32], gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 28, 35], gamma=0.1)
    d_scheduler = torch.optim.lr_scheduler.MultiStepLR(d_optimizer, milestones=[20, 28, 35], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    epoch = 40


    # start train model
    prev_time = datetime.now()
    best_metric = 0
    best_epoch = 0
    best_mat = None
    
    num_epochs = args.nepochs
    lambda_weight = np.ones([2, num_epochs+1])
    if args.weight == 'manual':
        lambda_weight[1,:] = args.miu
    elif args.weight == 'inter':
        for i in range(num_epochs+1):
            lambda_weight[0, i] = 1
            lambda_weight[1, i] = 0
            if i % args.inter == 0:
                lambda_weight[1, i] = 1
                
    for epoch in range(1, num_epochs+1):

        # train
        train(net, model1, model2, D, train_loader, criterion, num_classes, writer, lambda_weight, epoch)
        
        # test
        best_metric, best_epoch, best_mat = test(net, D, test_loader, criterion, num_classes, writer, best_metric, best_epoch, best_mat, lambda_weight, epoch)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        print(time_str)
        prev_time = cur_time
        
        scheduler.step()
        d_scheduler.step()

    print('Training finished!  Best Test metric: %.4f(epoch %d)' % (best_metric * 100, best_epoch))
    print('') 
    
    if args.dataset in ['RAF', 'FER2013']:
        print('Surprise Fear Disgust Happy Sad Anger Neutral')
    elif args.dataset in ['CK+', 'MMI', 'OULU']:
        print('Angry Surprise Disgust Fear Happy Sad Neutral')
    elif args.dataset == 'SFEW':
        print('Angry Disgust Fear Happy Neutral Sad Surprise')
    print(best_mat)
    writer.close()
