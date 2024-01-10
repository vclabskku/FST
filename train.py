import os
import warnings
import json
import random

import numpy as np
import torch
import torch.nn as nn
import timm
from sklearn.metrics import confusion_matrix, accuracy_score

from utils.opt import get_opts
from dataset.datasets import FSTDataset


def main(args):
    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)
    print("\nCUDA_VISIBLE_DEVICES: [{}]\n".format(os.environ["CUDA_VISIBLE_DEVICES"]), flush=True)
    
    # seed
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # checkpoint
    save_dir = "./ckpts/{}".format(args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # dataset
    print(">> Load the data...", flush=True)
    train_dataset = FSTDataset(args.dataset_dir, "train", args.meta_path, args.binary)
    val_dataset = FSTDataset(args.dataset_dir, "val", args.meta_path, args.binary)
    print("Train: {} (Total {:d})".format(train_dataset.id_counts, len(train_dataset)), flush=True)
    print("Val: {} (Total {:d})".format(val_dataset.id_counts, len(val_dataset)), flush=True)

    # dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False)
    print("<< Finished.\n", flush=True)

    # model
    print(">> Load the model...", flush=True)
    if args.binary:
        model=timm.create_model(args.model_name, pretrained=False, num_classes=2)
    else:
        model=timm.create_model(args.model_name, pretrained=False, num_classes=4)
    print("Model: {} from scratch".format(args.model_name))
    model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    model = model.cuda()
    print("<< Finished.\n")

    # optimizer & scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=len(train_loader)*args.step_size, 
                                                gamma=args.gamma)

    # train
    print(">> Train...", flush=True)
    best_acc = 0.
    for epoch in range(args.epochs):
        train(args, train_loader, model, criterion, optimizer, scheduler, epoch)
        result = val(args, val_loader, model, criterion)
        print("Epoch [{:02d}/{:02d}]\tVal Loss {:.3f}\tAcc {:.2f} {}"\
                .format(epoch+1, args.epochs, result['loss'], result['acc'], result['acc_list']), flush=True)
        if best_acc <= result['acc']:
            best_acc = result['acc']
            print("Save the epoch {:d}".format(epoch+1), flush=True)
            model_filename = os.path.join(save_dir, f'{args.exp_name}_best.pth')
            torch.save(
                {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                },
                model_filename
            ) 
        print("------------------------------------------------------------------------")
    print("<< Finished.\n")
    return


def train(args, train_loader, model, criterion, optimizer, scheduler, epoch):

    model.train()
    n_batch = len(train_loader)
    bs = args.batch_size

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        print("Epoch [{:02d}/{:02d}]\tBatch [{:02d}/{:02d}]\tTrain Loss {:.3f}"\
                .format(epoch+1, args.epochs, batch_idx+1, n_batch, loss.item()/bs), flush=True)

    return


def val(args, val_loader, model, criterion):

    model.eval()
    n_image = len(val_loader.dataset)
    n_batch = len(val_loader)
    bs = args.batch_size
    total_loss = 0.
    total_preds = torch.zeros(n_image)
    total_labels = torch.zeros(n_image)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.cuda()
            labels = labels.cuda()

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, preds = torch.max(logits, 1)
            total_idx = bs * batch_idx
            if batch_idx == n_batch-1:
                total_preds[total_idx:] = preds.detach().cpu()
                total_labels[total_idx:] = labels.detach().cpu()
            else:
                total_preds[total_idx:total_idx+bs] = preds.detach().cpu()
                total_labels[total_idx:total_idx+bs] = labels.detach().cpu()

    total_loss /= n_image
    matrix = confusion_matrix(total_labels, total_preds)
    acc = round(accuracy_score(total_labels, total_preds)*100, 2)
    acc_list = matrix.diagonal()/matrix.sum(axis=1)
    acc_list = [round(x*100, 2) for x in acc_list]

    return {'loss': total_loss,
            'acc': acc,
            'acc_list': acc_list}


if __name__ == '__main__':
    warnings.filterwarnings(action='ignore')
    torch.autograd.set_detect_anomaly(True)
    args = get_opts()
    print(json.dumps(vars(args), indent='\t'), flush=True)
    main(args)