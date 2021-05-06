import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from utils import AvgrageMeter, EERMeter, TotalAccuracyMeter
from tqdm import tqdm

def train(args, train_loader, val_loader, model, architect, criterion, lr, optimizer, epoch, writer_dict):
    losses = AvgrageMeter()
    total_acc_meter = TotalAccuracyMeter('Total accuracy')
    writer = writer_dict['writer']
    model.train()

    for step, (input, target) in enumerate(train_loader):
        global_steps = writer_dict['train_global_steps']

        # batch size
        n = input.size(0)

        input_search, target_search = next(iter(val_loader))
        # try:
        #     input_search, target_search = next(val_loader_iter)
        # except StopIteration:
        #     val_loader_iter = iter(val_loader)
        #     input_search, target_search = next(val_loader_iter)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        input_search = input_search.cuda(non_blocking=True)
        target_search = target_search.cuda(non_blocking=True)
    
        if epoch >= args.warm_up_epoch:
            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
    

        output, embeddings = model(input, args.is_mask)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), n)
        total_acc_meter.update(target, output)

        if step % args.report_freq == 0:
            logging.info('train %03d %e', step, losses.avg)
        
        writer.add_scalar('train_loss', losses.val, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1
        
    acc_per_epoch = total_acc_meter.get_accuracy()
    loss_per_epoch = losses.avg
    writer.add_scalar('arch_train_accuracy', acc_per_epoch, epoch)
    writer.add_scalar('arch_train_loss', loss_per_epoch, epoch)

    total_loss = losses.sum

    return acc_per_epoch*100, total_loss

def validate(dev_loader, model, criterion, epoch, writer_dict, validate_type):
    losses = AvgrageMeter()
    eermeter = EERMeter('EER', round_digits=4)
    total_acc_meter = TotalAccuracyMeter('Total accuracy')

    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(dev_loader):
            
            # batch size
            n = input.size(0)

            input = input.cuda(non_blocking=True).float()
            target = target.cuda(non_blocking=True)

            output = model(input, is_mask=False)
            output = model.forward_classifier(output)
            if validate_type == 'eval':
                eermeter.update(target, output)
            total_acc_meter.update(target, output)

            if criterion:
                loss = criterion(output, target)
                losses.update(loss.item(), n)

    acc_per_epoch = total_acc_meter.get_accuracy()
    loss_per_epoch = losses.avg
    total_loss = losses.sum
        
    if writer_dict:
        # get global step for tensorboard logging. We are not going to increase them in validation
        writer = writer_dict['writer']
        writer.add_scalar(validate_type + '_accuracy', acc_per_epoch*100, epoch)
        writer.add_scalar(validate_type + '_loss', loss_per_epoch, epoch)

    if validate_type == 'eval':
        eer_per_epoch = eermeter.get_eer()
        return acc_per_epoch*100, eer_per_epoch*100
    else:
        return acc_per_epoch*100, total_loss


def train_from_scratch(args, train_loader, model, optimizer, criterion, epoch, writer_dict):
    losses = AvgrageMeter()
    total_acc_meter = TotalAccuracyMeter('Total accuracy')
    writer = writer_dict['writer']

    model.train()
    for step, (input, target) in enumerate(train_loader):
        global_steps = writer_dict['train_global_steps']
        
        # batch size
        n = input.size(0)
        input = input.cuda(non_blocking=True)

        target = target.cuda(non_blocking=True)

        output, embeddings = model(input, args.is_mask)

        loss = criterion(output, target)
        losses.update(loss.item(), n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_acc_meter.update(target, output)

        if step % args.report_freq == 0:
            logging.info('train loss: %03d %e', step, losses.avg)
        writer.add_scalar('train_loss_step', losses.val, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

    acc_per_epoch = total_acc_meter.get_accuracy()
    loss_per_epoch = losses.avg
    writer.add_scalar('train_accuracy', acc_per_epoch*100, epoch)
    writer.add_scalar('train_loss', loss_per_epoch, epoch)

    total_loss = losses.sum

    return acc_per_epoch*100, total_loss


def evaluate(test_loader, model, comment):
    eermeter = EERMeter('EER', round_digits=4)
    total_acc_meter = TotalAccuracyMeter('Total accuracy')

    model.eval()
    fname_list = []
    key_list = []
    att_id_list = []
    key_list = []
    score_list = []

    with torch.no_grad():
        for step, (input, file_name, attack_id, target) in tqdm(enumerate(test_loader)):

            input = input.cuda(non_blocking=True).float()
            target = target.cuda(non_blocking=True)
            output = model(input, is_mask=False)
            output = model.forward_classifier(output)


            eermeter.update(target, output)
            total_acc_meter.update(target, output)

            fname_list.extend(list(file_name))
            key_list.extend(
            ['bonafide' if key == 1 else 'spoof' for key in target.tolist()])
            att_id_list.extend(list(attack_id))
            score_list.extend(output[:,1].tolist())

    save_path = 'score-' + comment + '.txt'
    with open(save_path, 'a') as fh:
        for f, s, k, cm in zip(fname_list, att_id_list, key_list, score_list):
            fh.write('{} {} {} {}\n'.format(f, s, k, cm))