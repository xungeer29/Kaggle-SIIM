import torch
import torch.nn.functional as F

from metrics import dice_metric, accuracy
from config import config

def train_one_epoch(model, data_loader, criterion, optimizer, lr_scheduler=None, device=None, epoch=None, vis=None):
    model.train()
    
    losses, dices = 0, 0
    dice_negs, dice_poss, num_negs, num_poss = 0, 0, 0, 0
    for i, (images, masks, labels, ids) in enumerate(data_loader):
        y_preds = model(images.to(device))

        loss = criterion(y_preds, masks.to(device))
        
        dice, dice_neg, dice_pos, num_neg, num_pos = dice_metric(y_preds, masks)

        if torch.cuda.device_count() > 1:
            loss = loss.mean() # mean() to average on multi-gpu.

        (loss/config.accumulation_steps).backward()
        if not config.Gradient_Accumulation:
            optimizer.step()
            optimizer.zero_grad()
        
        if config.Gradient_Accumulation and (((i+1)%config.accumulation_steps)==0 or (i+1)==len(data_loader)): 
            # optimizer the net
            optimizer.step()        # update parameters of net
            optimizer.zero_grad()   # reset gradient

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        losses += loss.item()
        dices += dice
        dice_negs += dice_neg
        dice_poss += dice_pos
        num_negs += num_neg
        num_poss += num_pos

        
        if (i+1) % config.interval == 0:
            mem = torch.cuda.max_memory_allocated()/1024./1024./1024.
            print('Epoch {}/{}, iter {}/{}, lr {:.5f}, loss {:.4f}, dice {}, dice_neg {}, dice_pos {}, num_neg {}, num_pos {}, memory {:.3f}Gb'.format(
                   epoch+1, config.num_epochs, i+1, len(data_loader), lr, losses/config.interval, dices/config.interval, 
                   dice_negs/config.interval, dice_poss/config.interval, num_negs/config.interval, num_poss/config.interval, mem))

            if vis is not None:
                vis.line(X=torch.FloatTensor([i+epoch*len(data_loader)]), Y=torch.FloatTensor([losses/config.interval]), 
                         win='loss', update='append' if (i+epoch*len(data_loader))>0 else None,
                         opts=dict(title='loss'), name='train_loss')
                vis.line(X=torch.FloatTensor([i+epoch*len(data_loader)]), Y=torch.FloatTensor([dices/config.interval]), 
                     win='dice', update='append' if (i+epoch*len(data_loader))>0 else None,
                     opts=dict(title='train_dice'), name='train_dice')
                vis.line(X=torch.FloatTensor([i+epoch*len(data_loader)]), Y=torch.FloatTensor([dice_negs/config.interval]), 
                     win='dice_neg', update='append' if (i+epoch*len(data_loader))>0 else None,
                     opts=dict(title='dice_neg'), name='train_dice_neg')
                vis.line(X=torch.FloatTensor([i+epoch*len(data_loader)]), Y=torch.FloatTensor([dice_poss/config.interval]), 
                     win='dice_pos', update='append' if (i+epoch*len(data_loader))>0 else None,
                     opts=dict(title='dice_pos'), name='train_dice_neg')
                vis.line(X=torch.FloatTensor([i+epoch*len(data_loader)]), Y=torch.FloatTensor([lr]), 
                     win='lr', update='append' if (i+epoch*len(data_loader))>0 else None,
                     opts=dict(title='lr'), name='lr')

            losses, dices = 0, 0
            dice_negs, dice_poss, num_negs, num_poss = 0, 0, 0, 0

        if lr_scheduler is not None:
            lr_scheduler.step()

    return model, optimizer
