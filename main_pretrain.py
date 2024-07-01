import argparse
import datetime
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import math
import logging
import sys
import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import seed_torch
from util.lr_sched import adjust_learning_rate
from data.LSDataset import LSDataset_pretrain
import model.models_mae as models_mae
from model.models_exp import PAME
from engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('pre-training', add_help=False)
    parser.add_argument('--dataset', type=str, default='sjtu', help='dataset with mos')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--warmup_epoch', type=int, default=5, help='epochs to warmup LR')
    parser.add_argument('--epoch', type=int, default=30, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--model', default='base', type=str, help='Name of model to pretrain')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--tensorBoard', action='store_true')
    parser.set_defaults(tensorBoard=False)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    # Model parameters
    
    parser.add_argument('--crop_size', default=224, type=int)
    parser.add_argument('--mask_ratio_ref', default=0.5, type=float)
    parser.add_argument('--mask_ratio_dist', default=0.5, type=float)

    # parser.add_argument('--norm_pix_loss', action='store_true',
    #                     help='Use (per-patch) normalized pixels as targets for computing loss')
    # parser.set_defaults(norm_pix_loss=False)
    
    return parser


def main(args):
    seed_torch(123)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = Path('./experiment')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('log/')
    log_dir.mkdir(exist_ok=True)
    output_dir = file_dir.joinpath('output/')
    output_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger('MAE')
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
    logger.info('Loading pretraining data...')
    print('Loading pretraining data...')
    pretrainDataset = LSDataset_pretrain(mode='pretrain',crop_size=args.crop_size)
    pretrainDataloader = DataLoader(pretrainDataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers,pin_memory=True)
    
    '''MODEL'''
    model = PAME()
    model = torch.nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
    model.cuda()

    eff_batch_size = args.batch_size * args.accum_iter
    eff_lr = args.lr * 256 / eff_batch_size

    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=eff_lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    
    '''PRETRAIN'''
    print('Start pretraining...'); logger.info('Start pretraining...')
    num_iter = len(pretrainDataloader)
    for epoch in range(args.epoch):
        model.train()
        optimizer.zero_grad()
        loss_epoch = 0
        for data_iter_step, (imgs,ref_imgs,mos) in tqdm(enumerate(pretrainDataloader), total=num_iter, leave=False):
            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % args.accum_iter == 0:
                adjust_learning_rate(optimizer, data_iter_step/num_iter + epoch, args)

            imgs = imgs.cuda()

            with torch.cuda.amp.autocast():
                loss_ref, loss_dist = model(imgs, mask_ratio_dist=args.mask_ratio_dist,
                                            ref_imgs=ref_imgs, mask_ratio_ref=args.mask_ratio_ref,
                                            mode='pretrain')

            loss = (loss_ref + loss_dist).mean()
            loss_value = loss.item()
            loss_epoch += (loss_value / num_iter)

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= args.accum_iter
            loss_scaler(loss, optimizer, parameters=model.parameters(),
                        update_grad=(data_iter_step + 1) % args.accum_iter == 0)
            if (data_iter_step + 1) % args.accum_iter == 0:
                optimizer.zero_grad()

            loss_value_reduce = misc.all_reduce_mean(loss_value)
            
            if args.tensorBoard and (data_iter_step + 1) % args.accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / num_iter + epoch) * 1000)
                log_writer.add_scalar('pretrain_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch_1000x)
    
        if epoch % 20 == 0 or epoch + 5 >= args.epoch:
            misc.save_model(args=args, checkpoint_dir=checkpoints_dir, model=model, 
                            optimizer=optimizer, epoch=epoch)
        
        logger.info('Epoch:{} lr:{:.6f} pretrain loss:{:.6f}'.format(epoch, optimizer.param_groups[0]["lr"],loss_epoch))
        print('Epoch:{} lr:{:.6f} pretrain loss:{:.6f}'.format(epoch, optimizer.param_groups[0]["lr"],loss_epoch))
    print('Pretrained models in {}'.format(checkpoints_dir.joinpath('checkpoint_{}.pth'.format(str(args.epoch-1)))))
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
    print(args)
