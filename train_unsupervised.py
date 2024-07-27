import os 
import random 
import argparse
import errno
import time
import copy 
from tqdm import tqdm 

from lib.models.models import TFMTSRL 
from lib.data.data import get_data_and_preprocess, get_data_and_preprocess_unsupervised, DatasetUnsupervised
from lib.utils.learning import AverageMeter
from lib.utils.utils import get_config

import numpy as np 
import torch 
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset




device = torch.device("cuda")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/unsupervised.yaml", help="Path to the config file.")
    parser.add_argument('--checkpoint', default='checkpoint_unsupervised', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    opts = parser.parse_args()
    return opts


def save_checkpoint(chk_path, epoch, lr, optimizer, scheduler, model, min_loss):
    model.eval()
              
    print("[INFO] : saving modelcheckpoint.")
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_loss' : min_loss, 
        'scheduler': scheduler.state_dict()
    }, chk_path)



def get_dataloader(args): 
     
    data = get_data_and_preprocess_unsupervised(args.csv_file, args.target, args.timesteps, args.train_split, args.val_split) 
    X_train_t, X_val_t, X_test_t, y_his_train_t, y_his_val_t, y_his_test_t  = data
    train_loader = DataLoader(
        DatasetUnsupervised(
            X = X_train_t, 
            timesteps = args.timesteps,
            y_known = None, 
            mask_r = args.mask_r, 
            lm = args.lm
        ), shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(
        DatasetUnsupervised(
            X = X_val_t, 
            timesteps = args.timesteps,
            y_known = None, 
            mask_r = args.mask_r, 
            lm = args.lm
        ), shuffle=True, batch_size=args.batch_size)
    test_loader = DataLoader(
        DatasetUnsupervised(
            X = X_test_t, 
            timesteps = args.timesteps,
            y_known = None, 
            mask_r = args.mask_r, 
            lm = args.lm
        ), shuffle=True, batch_size=args.batch_size)
    return train_loader, val_loader, test_loader 


def get_model(args): 
    model = TFMTSRL(
        w = args.timesteps, 
        m = args.num_exogenous, 
        d_dim = args.d_dim, 
        num_heads = args.num_heads, 
        num_layers = args.num_layers, 
        ff_hidden_dim = args.dim_ffw, 
        dropout = args.dropout, 
        out_dim = args.out_dim, 
        mode="unsupervised"
    )
    model.to(device)
    print(f"[INFO] : Running model on device : {device}")
    return model  


def get_criterion(args): 
    criterion = torch.nn.MSELoss() 
    criterion.to(device)
    return criterion 


def lr_lambda(iteration):
    return 0.9 ** (iteration // 10000)


def get_optimizer(args, model): 

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)    
    return optimizer, scheduler 


def train_epoch(args, opts, model, train_loader, criterion, optimizer, scheduler, losses, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        masked_input, target, mask_idx = data
        masked_input = masked_input.to(torch.float32)
        target = target.to(torch.float32) 
        mask_idx = mask_idx.to(torch.bool)
        masked_input = masked_input.to(device) # (bs, w, m)
        target = target.to(device)  # (bs, w, m)
        mask_idx = mask_idx.to(device) # (bs, w, m)

        preds = model(masked_input)  # (bs, w, m)        
        # masked_target = target[mask_idx]  
        # masked_preds = preds[mask_idx]  
        loss = criterion(preds, target)

        losses["train_MSE_loss"].update(loss.item(), preds.shape[0])
        loss.backward()
        optimizer.step()    
    
def validate_epoch(
    args, 
    opts, 
    model, 
    val_loader, 
    criterion, 
    losses, 
    epoch 
): 
    model.eval()

    with torch.no_grad(): 
        for data in val_loader: 
            masked_input, target, mask_idx = data
            masked_input = masked_input.to(torch.float32)
            target = target.to(torch.float32) 
            mask_idx = mask_idx.to(torch.bool)
            masked_input = masked_input.to(device) # (bs, w, m)
            target = target.to(device)  # (bs, w, m)
            mask_idx = mask_idx.to(device) # (bs, w, m)
            preds = model(masked_input)  # (bs, w, m)       

            loss = criterion(preds, target)
            losses["val_MSE_loss"].update(loss.item(), preds.shape[0]) 

    return model



def train_with_config(args, opts): 

    try: 
        os.mkdir(opts.checkpoint)
    except OSError as e: 
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
        
    train_writer = SummaryWriter(os.path.join(opts.checkpoint, "logs"))

    train_loader, val_loader, test_loader = get_dataloader(args)
    model = get_model(args)
    criterion = get_criterion(args)
    optimizer, scheduler = get_optimizer(args, model)

    min_loss = np.inf
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    for epoch in tqdm(range(args.epochs)): 
        print('Training epoch %d.' % epoch)

        losses = {}
        losses["train_MSE_loss"] = AverageMeter()
        losses["val_MSE_loss"] = AverageMeter()
        losses["test_MSE_loss"] = AverageMeter()
        train_epoch(args, opts, model, train_loader, criterion, optimizer, scheduler, losses, epoch) 
        model = validate_epoch(args, opts, model, val_loader, criterion, losses, epoch)
        
        # logs
        lr = optimizer.param_groups[0]['lr']
        train_writer.add_scalar("train_MSE_loss", losses["train_MSE_loss"].avg, epoch + 1)
        train_writer.add_scalar("val_MSE_loss", losses["val_MSE_loss"].avg, epoch + 1)
        train_writer.add_scalar("lr", lr, epoch + 1)
        chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch.bin')
        chk_path_best = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))

        save_checkpoint(chk_path_latest, epoch, lr, optimizer, scheduler, model, min_loss)
        if losses["val_MSE_loss"].avg < min_loss: 
            min_loss = losses["val_MSE_loss"].avg
            save_checkpoint(chk_path_best, epoch, lr, optimizer, scheduler, model, min_loss)



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__": 
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    train_with_config(args, opts)