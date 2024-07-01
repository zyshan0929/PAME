import argparse
import datetime
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import torch
torch.set_num_threads(4)
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import math
import logging
import sys
import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
import util.misc as misc
import pandas as pd
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import seed_torch
from util.lr_sched import adjust_learning_rate
from data.LSDataset import LSDataset_pretrain, LSDataset
from util.pos_embed import interpolate_pos_embed
import model.models_mae as models_mae
from model.models_exp import PAME, Fusion
from engine_pretrain import train_one_epoch
from util.loss import L2RankLoss
from util.logistic_4_fitting import logistic_4_fitting


def get_args_parser():
    parser = argparse.ArgumentParser('finetune', add_help=False)
    parser.add_argument('--dataset', type=str, default='ls', help='dataset for finetuning')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--warmup_epoch', type=int, default=10, help='epochs to warmup LR')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--model', default='base', type=str, help='Name of model to pretrain')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--tensorBoard', action='store_true')
    parser.set_defaults(tensorBoard=False)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--num_view', type=int, default=6)
    parser.add_argument('--crop_size', default=224, type=int)
    
    return parser


def main(args):
    seed_torch(123)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = Path('./experiment')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('log/')
    log_dir.mkdir(exist_ok=True)
    output_dir = file_dir.joinpath('output/')
    output_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger('FINETUNE')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/' + 'log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('Parameters...')
    logger.info(args)
    if args.tensorBoard:
        log_writer = SummaryWriter(log_dir=str(log_dir))
    print('Parameters...\n', args)
    
    '''DATA LOADING'''
    logger.info('Loading finetuning data...')
    print('Loading finetuning data...')
    trainDataset = LSDataset(mode='train',crop_size=args.crop_size, num_view=args.num_view)
    trainDataloader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers,pin_memory=True)
    testDataset = LSDataset(mode='test', crop_size=args.crop_size)
    testDataloader = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
    
    '''MODEL'''
    model = PAME()
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        checkpoint_model = checkpoint['model']
        model.load_state_dict(checkpoint_model, strict=False)
    model_fusion = Fusion()
    model, model_fusion = model.cuda(), model_fusion.cuda()
    
    '''OPTIMIZER'''
    optimizer = torch.optim.Adam(list(model.parameters())+list(model_fusion.parameters()), lr=args.lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    criterion = L2RankLoss()
    
    '''TRAIN'''
    print('Start training'); logger.info('Start training')
    num_iter_train, num_iter_test = len(trainDataloader), len(testDataloader)
    for epoch in range(args.epoch):
        model.train(); model_fusion.train()
        optimizer.zero_grad()
        loss_epoch_train = 0
        for data_iter_step, (imgs,mos) in tqdm(enumerate(trainDataloader), total=num_iter_train, leave=False):
            # we use a per iteration (instead of per epoch) lr scheduler
            optimizer.zero_grad()

            imgs, mos = imgs.cuda(), mos.cuda()
            B, view_length, C, H, W = imgs.shape
            latent_ref, latent_dist = model(imgs.view(-1, C, H, W), mode='finetune')
            latent_ref, latent_dist = latent_ref.view(B, view_length, -1), latent_dist.view(B, view_length,-1)
            pred_mos = model_fusion(latent_ref, latent_dist)

            loss = criterion(pred_mos, mos)
            loss_value = loss.item()
            loss_epoch_train += loss_value

            loss.backward()
            optimizer.step()
        
        logger.info('Epoch:{} lr:{:.6f} train loss:{:.6f}'.format(epoch, optimizer.param_groups[0]["lr"],loss_epoch_train/num_iter_train))
        print('Epoch:{} lr:{:.6f} train loss:{:.6f}'.format(epoch, optimizer.param_groups[0]["lr"],loss_epoch_train/num_iter_train))
        scheduler.step()

        '''TEST'''
        with torch.no_grad():
            loss_epoch_test = 0
            model.eval(); model_fusion.eval()
            pred_mos_list, mos_list = [], []
            for data_iter_step, (imgs, mos) in tqdm(enumerate(testDataloader),total=num_iter_test,leave=False):
                imgs, mos = imgs.cuda(), mos.cuda()
                B, view_length, C, H, W = imgs.shape
                latent = model(imgs.view(-1, C, H, W), mode='finetune').view(B,view_length,-1)
                pred_mos = model_fusion(latent).view_as(mos)

                loss = criterion(pred_mos, mos)
                loss_epoch_test += loss.item()
                pred_mos = pred_mos.data.cpu().view_as(mos).numpy()
                mos = mos.data.cpu().numpy()
                pred_mos_list.extend(list(pred_mos))
                mos_list.extend(list(mos))
            _, __, pred_mos_list = logistic_4_fitting(pred_mos_list, mos_list)
            pred_mos_series, mos_series = pd.Series(pred_mos_list), pd.Series(mos_list)
            srocc = pred_mos_series.corr(mos_series, method="spearman")
            plcc = pred_mos_series.corr(mos_series, method="pearson")
            rmse = ((pred_mos_series-mos_series)**2).mean() ** .5
            logger.info('Epoch:{} test loss: {:.3f} SROCC: {:.3f} PLCC: {:.3f} RMSE: {:.3f}'.format(epoch,loss_epoch_test/num_iter_test,srocc,plcc,rmse))
            print('Epoch:{} test loss: {:.3f} SROCC: {:.3f} PLCC: {:.3f} RMSE: {:.3f}\n'.format(epoch,loss_epoch_test/num_iter_test,srocc,plcc,rmse))
        print('Experiment files in {}'.format(str(file_dir)))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
    print(args)
