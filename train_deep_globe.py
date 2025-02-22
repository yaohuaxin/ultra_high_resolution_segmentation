#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import os
import pathlib
import numpy as np
import random
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from dataset.deep_globe import classToRGB
from dataset.PAIP2019 import PAIP2019
from utils.loss import CrossEntropyLoss2d, SoftCrossEntropyLoss2d, FocalLoss
from utils.lovasz_losses import lovasz_softmax
from utils.lr_scheduler import LR_Scheduler
from tensorboardX import SummaryWriter
from helper import create_model_load_weights, get_optimizer, Trainer, Evaluator, collate, collate_test
from option import Options

# from torch.nn.parallel import DistributedDataParallel as DDP
# Why use apex: https://github.com/pytorch/pytorch/issues/13273
# Tested with https://github.com/NVIDIA/apex.git,
#     Commit ID: 665b2dd7dc9d5129d7541bad612c1d86ba4b6818
from apex.parallel import DistributedDataParallel as DDP

#
# System wide configurations
#
np.set_printoptions(linewidth=200)
# torch.cuda.synchronize()
# torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Enable LMS (Large model support)
torch.cuda.set_enabled_lms(True)

#
# Application Parameters
#
args = Options().parse()

task_name   = args.task_name
n_class     = args.n_class
data_path   = args.data_path
image_level = args.image_level
model_path  = args.model_path
log_path    = args.log_path
data_loader_worker = args.data_loader_worker

# Running mode definations: 
#     1: train global; 
#     2: train local from global; 
#     3: train global from local
mode       = args.mode
evaluation = args.evaluation
test       = evaluation and False

##### sizes are (w, h) ##############################
# make sure margin / 32 is over 1.5 AND size_g is divisible by 4
size_g         = (args.size_g, args.size_g) # resized global image
size_p         = (args.size_p, args.size_p) # cropped local patch size
batch_size     = args.batch_size
sub_batch_size = args.sub_batch_size        # batch size for train local patches

num_epochs     = args.num_epochs
learning_rate  = args.lr
lamb_fmreg     = args.lamb_fmreg

path_g   = os.path.join(model_path, args.path_g)
path_g2l = os.path.join(model_path, args.path_g2l)
path_l2g = os.path.join(model_path, args.path_l2g)

distributed = int(os.environ["WORLD_SIZE"]) > 1 if "WORLD_SIZE" in os.environ else False
world_size  = int(os.environ["WORLD_SIZE"]) if distributed else 1

if (not distributed):
    print("Please run in distributted mode.")
    exit(1)

local_rank = args.local_rank
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend="nccl", init_method="env://")

if (torch.distributed.get_rank()==0):
    print("============Model Information (Begin)==========================")
    print("Running in distributted:", distributed)
    print("Task name   :", task_name)
    print("Running mode:", mode, "; evaluation:", evaluation, "; test:", test)
    print("size_g:", size_g, "; batch_size  :", batch_size, "; size_p", size_p, "; sub_batch_size:", sub_batch_size)
    print("============Model Information (End)============================")

    if not os.path.isdir(model_path):
        pathlib.Path(model_path).mkdir(parents=True)
    if not os.path.isdir(log_path):
        pathlib.Path(log_path).mkdir(parents=True)
    
    pass

# to synchronize start of time
torch.distributed.barrier()

# Explicitly setting seed to make sure that models created in all processes 
# start from same random weights and biases.
# https://pytorch.org/docs/stable/notes/randomness.html
seed_tensor = torch.tensor(0, dtype=torch.float32, device=torch.device("cuda"))
if torch.distributed.get_rank() == 0:
    # seed = int(time.time())
    # random master seed, random.SystemRandom() uses /dev/urandom on Unix
    master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)    
    seed_tensor = torch.tensor(master_seed, dtype=torch.float32, device=torch.device("cuda"))

torch.distributed.broadcast(seed_tensor, 0)
master_seed = int(seed_tensor.item())

torch.manual_seed(master_seed)
torch.cuda.manual_seed(master_seed)    
np.random.seed(master_seed)
random.seed(master_seed)
    
#
# Dataset and Dataloader
#
ids_images = []
for phase_fold in os.listdir(data_path):
    if os.path.isdir(os.path.join(data_path, phase_fold)):
        for image_fold in os.listdir(os.path.join(data_path, phase_fold)):
            if os.path.isdir(os.path.join(data_path, phase_fold, image_fold)):
                ids_images.append(os.path.join(phase_fold, image_fold))

ids_train = ids_images[0:-8]
ids_val   = ids_images[-8:]
ids_test  = ids_images[-8:]

# print("==== ids_train: ", ids_train)
# print("==== ids_val: ", ids_val)
# print("==== ids_test: ", ids_test)

dataset_train = PAIP2019(data_path, ids_train, label=True, transform=True, image_level=image_level)
sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train, world_size, local_rank)
dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, 
                                               num_workers=data_loader_worker, collate_fn=collate, 
                                               shuffle=False, pin_memory=True, sampler=sampler_train)

dataset_val = PAIP2019(data_path, ids_val, label=True, image_level=image_level)
sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val, world_size, local_rank)
dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, 
                                             num_workers=data_loader_worker, collate_fn=collate, 
                                             shuffle=False, pin_memory=True, sampler=sampler_val)

dataset_test = PAIP2019(data_path, ids_test, label=False, image_level=image_level)
sampler_test = torch.utils.data.distributed.DistributedSampler(dataset_test, world_size, local_rank)
dataloader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size,
                                              num_workers=data_loader_worker, collate_fn=collate_test,
                                              shuffle=False, pin_memory=True, sampler=sampler_test)

#
# Create the model
#
model, global_fixed = create_model_load_weights(n_class, mode, evaluation, path_g=path_g, 
                                                path_g2l=path_g2l, path_l2g=path_l2g, local_rank=local_rank)
'''
if (torch.distributed.get_rank()==0):
    for name, param in model.named_parameters(): 
        print(name, True if param.grad is not None else False)
'''

# model_ddp = DDP(model, device_ids=[local_rank], output_device=local_rank)
model_ddp = DDP(model)

optimizer = get_optimizer(model_ddp, mode, learning_rate=learning_rate)
scheduler = LR_Scheduler('poly', learning_rate, num_epochs, len(dataloader_train))

criterion1 = FocalLoss(gamma=3)
criterion2 = nn.CrossEntropyLoss()
criterion3 = lovasz_softmax
criterion  = lambda x,y: criterion1(x, y)
# criterion = lambda x,y: 0.5*criterion1(x, y) + 0.5*criterion3(x, y)
mse = nn.MSELoss()

if not evaluation:
    writer = SummaryWriter(logdir=log_path + task_name + ".rank_" + str(torch.distributed.get_rank()))
    f_log = open(log_path + task_name + ".rank_" + str(torch.distributed.get_rank()) + ".log", 'w')

trainer   = Trainer(criterion, optimizer, n_class, size_g, size_p, sub_batch_size, mode, lamb_fmreg)
evaluator = Evaluator(n_class, size_g, size_p, sub_batch_size, mode, test)

best_pred = 0.0
if (torch.distributed.get_rank()==0):
    print("start training......")

for epoch in range(num_epochs):
    trainer.set_train(model_ddp)
    optimizer.zero_grad()
    
    # Huaxin: maybe should call scheduler outside of training epoch
    # scheduler(optimizer, epoch, epoch, best_pred)
    
    tbar = tqdm(dataloader_train); train_loss = 0
    for i_batch, sample_batched in enumerate(tbar):
        images, labels = sample_batched['image'], sample_batched['label'] # PIL images
        #print(images[0].size, labels[0].size,)
        
        if evaluation: 
            break
        
        scheduler(optimizer, i_batch, epoch, best_pred)
        
        loss = trainer.train(sample_batched, model_ddp, global_fixed)
        train_loss += loss.item()
        score_train, score_train_global, score_train_local = trainer.get_scores()
        if mode == 1:
            tbar.set_description('Train loss: %.3f; global mIoU: %.3f' % (train_loss / (i_batch + 1), np.mean(np.nan_to_num(score_train_global["iou"]))))
            pass
        else:
            tbar.set_description('Train loss: %.3f; agg mIoU: %.3f' % (train_loss / (i_batch + 1), np.mean(np.nan_to_num(score_train["iou"]))))
            pass

    score_train, score_train_global, score_train_local = trainer.get_scores()
    trainer.reset_metrics()
    # torch.cuda.empty_cache()

    #print("Epoch:", epoch, "training finished on rank:", torch.distributed.get_rank())
    torch.distributed.barrier()
    #print("Will start next training/testing on rank  :", torch.distributed.get_rank())

    if (epoch+1) % 5 == 0:
        with torch.no_grad():
            model_ddp.eval()
            print("\n"+"evaluating after epoch: ", epoch)

            if test:
                tbar = tqdm(dataloader_test)
            else:
                tbar = tqdm(dataloader_val)

            for i_batch, sample_batched in enumerate(tbar):
                predictions, predictions_global, predictions_local = evaluator.eval_test(sample_batched, model_ddp, global_fixed)
                score_val, score_val_global, score_val_local = evaluator.get_scores()
                if mode == 1: 
                    tbar.set_description('global mIoU: %.3f' % (np.mean(np.nan_to_num(score_val_global["iou"]))))
                else:
                    tbar.set_description('agg mIoU: %.3f' % (np.mean(np.nan_to_num(score_val["iou"]))))
                
                images = sample_batched['image']
                if not test:
                    labels = sample_batched['label'] # PIL images

                if test:
                    if not os.path.isdir("./prediction/"): os.mkdir("./prediction/")
                    for i in range(len(images)):
                        if mode == 1:
                            transforms.functional.to_pil_image(classToRGB(predictions_global[i]) * 255.).save("./prediction/" + sample_batched['id'][i] + "_mask.png")
                        else:
                            transforms.functional.to_pil_image(classToRGB(predictions[i]) * 255.).save("./prediction/" + sample_batched['id'][i] + "_mask.png")

                if not evaluation and not test:
                    if i_batch * batch_size + len(images) > (epoch % len(dataloader_val)) and i_batch * batch_size <= (epoch % len(dataloader_val)):
                        writer.add_image('image', transforms.ToTensor()(images[(epoch % len(dataloader_val)) - i_batch * batch_size]), epoch)
                        if not test:
                            writer.add_image('mask', classToRGB(np.array(labels[(epoch % len(dataloader_val)) - i_batch * batch_size])) * 255., epoch)
                        if mode == 2 or mode == 3:
                            writer.add_image('prediction', classToRGB(predictions[(epoch % len(dataloader_val)) - i_batch * batch_size]) * 255., epoch)
                            writer.add_image('prediction_local', classToRGB(predictions_local[(epoch % len(dataloader_val)) - i_batch * batch_size]) * 255., epoch)
                        writer.add_image('prediction_global', classToRGB(predictions_global[(epoch % len(dataloader_val)) - i_batch * batch_size]) * 255., epoch)

            # torch.cuda.empty_cache()

            if torch.distributed.get_rank() == 0:
                # All processes should see same parameters as they all start from same
                # random parameters and gradients are synchronized in backward passes.
                # Therefore, saving it in one process is sufficient.
                if mode == 1:
                    if not (test or evaluation): torch.save(model_ddp.state_dict(), os.path.join(model_path, path_g+"."+str(epoch)))
                elif mode == 2:
                    if not (test or evaluation): torch.save(model_ddp.state_dict(), os.path.join(model_path, path_g2l+"."+str(epoch)))
                elif mode == 3:
                    if not (test or evaluation): torch.save(model_ddp.state_dict(), os.path.join(model_path, path_l2g+"."+str(epoch)))
                else:
                    pass
            
            # Use a barrier() to make sure that below work started after rank 0 finish saving the states.
            torch.distributed.barrier()
            
            if test:
                break
            else:
                score_val, score_val_global, score_val_local = evaluator.get_scores()
                evaluator.reset_metrics()
                if mode == 1:
                    if np.mean(np.nan_to_num(score_val_global["iou"][1:])) > best_pred: best_pred = np.mean(np.nan_to_num(score_val_global["iou"][1:]))
                    # if np.mean(np.nan_to_num(score_val_global["iou"])) > best_pred: best_pred = np.mean(np.nan_to_num(score_val_global["iou"]))
                else:
                    if np.mean(np.nan_to_num(score_val["iou"][1:])) > best_pred: best_pred = np.mean(np.nan_to_num(score_val["iou"][1:]))
                    # if np.mean(np.nan_to_num(score_val["iou"])) > best_pred: best_pred = np.mean(np.nan_to_num(score_val["iou"]))
                log = "====================================================================================================\n"
                log = log + 'epoch [{}/{}]           IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_train["iou"][1:])), np.mean(np.nan_to_num(score_val["iou"][1:]))) + "\n"
                log = log + 'epoch [{}/{}] Local  -- IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_train_local["iou"][1:])), np.mean(np.nan_to_num(score_val_local["iou"][1:]))) + "\n"
                log = log + 'epoch [{}/{}] Global -- IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_train_global["iou"][1:])), np.mean(np.nan_to_num(score_val_global["iou"][1:]))) + "\n"
                # log = log + 'epoch [{}/{}] IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_train["iou"])), np.mean(np.nan_to_num(score_val["iou"]))) + "\n"
                # log = log + 'epoch [{}/{}] Local  -- IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_train_local["iou"])), np.mean(np.nan_to_num(score_val_local["iou"]))) + "\n"
                # log = log + 'epoch [{}/{}] Global -- IoU: train = {:.4f}, val = {:.4f}'.format(epoch+1, num_epochs, np.mean(np.nan_to_num(score_train_global["iou"])), np.mean(np.nan_to_num(score_val_global["iou"]))) + "\n"
                log = log + "train       : " + str(score_train["iou"]) + "\n"
                log = log + "val         : " + str(score_val["iou"]) + "\n"
                log = log + "Local train : " + str(score_train_local["iou"]) + "\n"
                log = log + "Local val   : " + str(score_val_local["iou"]) + "\n"
                log = log + "Global train: " + str(score_train_global["iou"]) + "\n"
                log = log + "Global val  : " + str(score_val_global["iou"]) + "\n"
                log += "====================================================================================================\n"
                print(log)
                if evaluation: break

                f_log.write(log)
                f_log.flush()
                
                if mode == 1:
                    writer.add_scalars('IoU', {'train iou': np.mean(np.nan_to_num(score_train_global["iou"][1:])), 'validation iou': np.mean(np.nan_to_num(score_val_global["iou"][1:]))}, epoch)
                    # writer.add_scalars('IoU', {'train iou': np.mean(np.nan_to_num(score_train_global["iou"])), 'validation iou': np.mean(np.nan_to_num(score_val_global["iou"]))}, epoch)
                else:
                    writer.add_scalars('IoU', {'train iou': np.mean(np.nan_to_num(score_train["iou"][1:])), 'validation iou': np.mean(np.nan_to_num(score_val["iou"][1:]))}, epoch)
                    # writer.add_scalars('IoU', {'train iou': np.mean(np.nan_to_num(score_train["iou"])), 'validation iou': np.mean(np.nan_to_num(score_val["iou"]))}, epoch)

if not evaluation: f_log.close()
