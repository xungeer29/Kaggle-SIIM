import torch

from metrics import dice_metric, accuracy

def do_valid(net, valid_loader, criterion, epoch, device, vis=None):
    net.eval().to(device)
    losses, dices, dice_negs, dice_poss, num_negs, num_poss = 0, 0, 0, 0, 0, 0
    nums = len(valid_loader)

    with torch.no_grad():
        for i, (im, mask, label, id) in enumerate(valid_loader):
            im = im.to(device)
            mask = mask.to(device)
            label = label.to(device)
        
            logit_seg = net(im)
            prob_seg = torch.sigmoid(logit_seg)
            loss = criterion(logit_seg, mask)

            dice, dice_neg, dice_pos, num_neg, num_pos = dice_metric(logit_seg, mask)
            # acc_top1 = accuracy(logit_cls.data, label.data, topk=(1,))

            losses += loss
            dices += dice
            dice_negs += dice_neg
            dice_poss += dice_pos
            num_negs += num_neg
            num_poss += num_pos

    print('Validation: loss {:.3f}, dice {}, neg dic {}, pos dice {}, num_neg {}, num_pos {}'.format(
            losses/nums, dices/nums, dice_negs/nums, dice_poss/nums, num_negs/nums, num_poss/nums))
    if vis is not None:
        vis.line(X=torch.FloatTensor([(epoch+1)*nums]), Y=torch.FloatTensor([losses/nums]), 
                 win='segmentation loss', update='append' if epoch>0 else None, name='val_loss')
        vis.line(X=torch.FloatTensor([(epoch+1)*nums]), Y=torch.FloatTensor([dices/nums]), 
                 win='dice', update='append' if epoch>0 else None, name='val_dice')
        vis.line(X=torch.FloatTensor([(epoch+1)*nums]), Y=torch.FloatTensor([dice_negs/nums]), 
                 win='dice_neg', update='append' if epoch>0 else None, name='val_dice_neg')
        vis.line(X=torch.FloatTensor([(epoch+1)*nums]), Y=torch.FloatTensor([dice_poss/nums]), 
                 win='dice_pos', update='append' if epoch>0 else None, name='val_dice_pos')