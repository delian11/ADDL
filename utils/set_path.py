import os
import shutil
from tensorboardX import SummaryWriter

import sys 
sys.path.append("..") 
from main import args

def set_writer(writer_path):
    if os.path.exists(writer_path):
        shutil.rmtree(writer_path)
    else:
        os.mkdir(writer_path)
    print('==> Writer saved to {}'.format(writer_path))
    writer = SummaryWriter(writer_path)
    return writer


def set_path(root):
    # set model save path
    global save_dir, writer
    save_root = os.path.join(root, 'Model/DDL')
    if not os.path.isdir(save_root):
        os.mkdir(save_root)
    save_root = os.path.join(save_root, args.dataset)
    if not os.path.isdir(save_root):
        os.mkdir(save_root)
    if args.weight == 'manual':
        save_dir = 'lr' + str(args.lr) + '_miu' + str(args.miu)
    elif args.weight == 'inter':
        save_dir = 'lr' + str(args.lr) + '_inter' + str(args.inter)
    if args.dataset in ['CK+', 'MMI', 'OULU']:
        save_dir = save_dir + '_fold' + str(args.fold)
    save_dir = os.path.join(save_root, save_dir)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    print('==> Model saved to {}.'.format(save_dir))

    # set writer path
    log_root = 'train_log'
    if not os.path.isdir(log_root):
        os.mkdir(log_root)
    log_root = os.path.join(log_root, args.dataset)
    if not os.path.isdir(log_root):
        os.mkdir(log_root)
    log_dir = 'lr' + str(args.lr) + '_inter' + str(args.inter)
    
    if args.dataset in ['CK+', 'MMI', 'OULU']:
        log_dir = log_dir + '_fold' + str(args.fold)
    writer = set_writer(os.path.join(log_root, log_dir))

    #set feature picture path
    global plot_dir
    if args.plot:
        pic_dir = 'pics'
        plot_dir = os.path.join(pic_dir, 'features')
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        print('==> Feature picture saved to {}.'.format(plot_dir))
        return save_dir, writer, plot_dir
    else:
        return save_dir, writer