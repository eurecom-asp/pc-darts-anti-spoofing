import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from ASVRawDataset import ASVRawDataset
from model import Network
from architect import Architect
from pathlib import Path
from functions import train_from_scratch, validate
from utils import Genotype

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ASVSpoof2019 model')
    parser.add_argument('--data', type=str, default='/path/to/your/LA', 
                    help='location of the data corpus')          
    parser.add_argument('--valid_freq', type=int, default=1, help='validate frequency')
    parser.add_argument('--report_freq', type=int, default=1000, help='report frequency in training')
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--init_channels', type=int, default=16)
    parser.add_argument('--arch', type=str, help='the searched architecture')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--nfft', type=int, default=1024, help='number of FFT point')
    parser.add_argument('--hop', type=int, default=4, help='number of hop size (nfft//hop)')
    parser.add_argument('--nfilter', type=int, default=70, help='number of linear filter')
    parser.add_argument('--num_ceps', type=int, default=20, help='LFCC dimention before deltas')
    parser.add_argument('--log', dest='is_log', action='store_true', help='whether use log(STFT)')
    parser.add_argument('--no-log', dest='is_log', action='store_false', help='whether use log(STFT)')
    parser.add_argument('--mask', dest='is_mask', action='store_true', help='whether use freq mask')
    parser.add_argument('--no-mask', dest='is_mask', action='store_false', help='whether use freq mask')
    parser.add_argument('--cmvn', dest='is_cmvn', action='store_true', help='whether zero-mean std')
    parser.add_argument('--no-cmvn', dest='is_cmvn', action='store_false', help='whether zero-mean std')
    parser.add_argument('--frontend', type=str, help='select frontend')
    parser.add_argument('--sr', type=int, default=16000, help='default sampling rate')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_min', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--comment', type=str, default='EXP', help='Comment to describe the saved mdoel')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')

    parser.set_defaults(is_log=True)
    parser.set_defaults(is_mask=False)
    parser.set_defaults(is_cmvn=False)

    args = parser.parse_args()
    args.comment = 'train-{}-{}'.format(args.comment, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.comment, scripts_to_save=glob.glob('*.py'))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.comment, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    model_save_path = os.path.join(args.comment, 'models')
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    train_protocol = 'ASVspoof2019.LA.cm.train.trn.txt'
    dev_protocol = 'ASVspoof2019.LA.cm.dev.trl.txt'


    if args.frontend == 'spec':
        front_end = 'Spectrogram'
        logging.info('-----Using STFT frontend-----')
    elif args.frontend == 'lfcc':
        front_end = 'LFCC'
        logging.info('-----Using LFCC frontend-----')
    elif args.frontend == 'lfb':
        front_end = 'LFB'
        logging.info('-----Using LFB frontend-----')
    else:
        print('*************No frontend selected*********************')

    OUTPUT_CLASSES = 2
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logging.info("args = %s", args)
    
    device = 'cuda'
    weight = torch.FloatTensor([1.0, 9.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    criterion = criterion.cuda()

    genotype = eval(args.arch)
    model = Network(args.init_channels, args.layers, args, OUTPUT_CLASSES, genotype, front_end)
    model = model.to(device)
    # logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    logging.info("param size = %fM", utils.count_parameters(model))

    train_dataset = ASVRawDataset(Path(args.data), 'train', train_protocol)
    dev_dataset = ASVRawDataset(Path(args.data), 'dev', dev_protocol)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    dev_loader = torch.utils.data.DataLoader(
        dataset=dev_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )


    optimizer = torch.optim.Adam(
        model.parameters(),
        args.lr,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.num_epochs), eta_min=args.lr_min)

    begin_epoch = 0
    best_acc = 85
    writer_dict = {
        'writer': SummaryWriter(args.comment),
        'train_global_steps': begin_epoch * len(train_loader),
        'valid_global_steps': begin_epoch // args.valid_freq,
    }

    for epoch in range(args.num_epochs):
        lr = scheduler.get_last_lr()[0]
        logging.info('Epoch: %d lr: %e', epoch, lr)

        model.drop_path_prob = args.drop_path_prob * epoch / args.num_epochs
        
        train_acc, train_loss = train_from_scratch(args, train_loader, model, optimizer, criterion, epoch, writer_dict)
        logging.info('train_loss %f', train_loss)
        logging.info('train_acc %f', train_acc)

        # validation
        if epoch % args.valid_freq == 0:
            dev_acc, dev_loss = validate(dev_loader, model, criterion, epoch, writer_dict, validate_type='dev')
            logging.info('dev_loss %f', dev_loss)
            logging.info('dev_acc %f', dev_acc)

        torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))
        scheduler.step()