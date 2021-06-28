import argparse
import numpy as np
import os
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from models.model import Network
from func.functions import evaluate
from ASVDataloader.ASVRawTest import ASVRawTest
from utils.utils import Genotype


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ASVSpoof2019 model')
    parser.add_argument('--data', type=str, default='/path/to/your/LA', 
                    help='location of the data')   
    parser.add_argument('--model', type=str, default='/path/to/your/saved/models/epoch_x.pth')
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--init_channels', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=1)
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
    parser.add_argument('--arch', type=str, help='the searched architecture')
    parser.add_argument('--comment', type=str, default='exp')
    parser.add_argument('--eval', type=str, default='e', help='to use e-eval or d-dev partition')

    parser.set_defaults(is_log=True)
    parser.set_defaults(is_mask=False)
    parser.set_defaults(is_cmvn=False)

    args = parser.parse_args()
    OUTPUT_CLASSES = 2
    checkpoint = torch.load(args.model)
    genotype = eval(args.arch)

    if args.frontend == 'spec':
        front_end = 'Spectrogram'
    elif args.frontend == 'lfcc':
        front_end = 'LFCC'
    elif args.frontend == 'lfb':
        front_end = 'LFB'


    model = Network(args.init_channels, args.layers, args, OUTPUT_CLASSES, genotype, front_end)
    model.drop_path_prob = 0.0

    if args.eval == 'e':
        eval_protocol = 'ASVspoof2019.LA.cm.eval.trl.txt'
        eval_dataset = ASVRawTest(Path(args.data), 'eval', eval_protocol)
    elif args.eval == 'd':
        print('*'*50)
        print('using dev protocol...')
        eval_protocol = 'ASVspoof2019.LA.cm.dev.trl.txt'
        eval_dataset = ASVRawTest(Path(args.data), 'dev', eval_protocol)
        print('*'*50)


    model = model.cuda()
    model.load_state_dict(checkpoint)

    eval_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    evaluate(eval_loader, model, args.comment)
    print('Done')
