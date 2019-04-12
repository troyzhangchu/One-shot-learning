import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms
import torchvision.datasets as datasets
import utils.loc as loc
import torch
from torch.utils.data import DataLoader
import random
import os
from PIL import Image
from utils.AverageMeter import AverageMeter
from utils.metric import accuracy,update_class_acc
import utils.get_data as gd

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

"""Pretrain the basic classifors and features extracter """

torch.manual_seed(1)

# Parameters
EPOCH = 5
BATCH_SIZE = 48
LR = 0.0001           # learning rate


# Resnet50
resnet = models.resnet50(pretrained=False).cuda()
optimizer = torch.optim.Adam(resnet.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
for epoch in range(EPOCH):
    print("Epoch {}".format(epoch))
    for step, sample in enumerate(gd.train_loader):   # gives batch data
        s = Variable(sample[0].cuda())
        output = resnet(s)               # output
        loss = loss_func(output, sample[1].cuda())   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        print("Iter {}".format(step))

# save model
new_path = os.path.join(loc.resnet_dir, "version")
if not os.path.exists(new_path):
    os.mkdir(new_path)
torch.save(resnet, os.path.join(new_path, "model_best.pth.tar"))

# evaluate on val
top1 = AverageMeter()
top5 = AverageMeter()
accu_20 = []
for i in range(80):
    accu_20.append(AverageMeter())

for step, sample in enumerate(gd.val_loader):
    weight = sample[0].shape[0]
    s = Variable(sample[0].cuda())
    pre = resnet(s)
    prec1, prec5 = accuracy(pre.data, sample[1].cuda(), topk=(1, 5))
    top1.update(prec1[0], n=weight)
    top5.update(prec5[0], n=weight)
    update_class_acc(accu_20, pre.data, sample[1].cuda())


    print("Step: {step}, top1: {top1.avg:.3f}({top1.val:.3f}), "
          "top5: {top5.avg:.3f}({top5.val:.3f})".format(step=step, top1=top1, top5=top5))

# for k, j in enumerate(accu_20):
#     print("{k}: {top1.avg:.3f}({top1.val:.3f}), ".format(k=k, top1=j))
