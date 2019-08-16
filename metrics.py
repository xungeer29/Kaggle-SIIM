import numpy as np
import torch

def dice_metric(logit, truth, threshold=0.5, reduction='none'):
    batch_size = len(truth)
    logit = logit.cpu()
    truth = truth.cpu()

    with torch.no_grad():
        logit = logit.view(batch_size,-1)
        truth = truth.view(batch_size,-1)
        assert(logit.shape==truth.shape)

        probability = torch.sigmoid(logit)
        p = (probability>threshold).float()
        t = (truth>0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum==0)
        pos_index = torch.nonzero(t_sum>=1)
        #print(len(neg_index), len(pos_index))


        dice_neg = (p_sum == 0).float()
        dice_pos = 2* (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice     = torch.cat([dice_pos,dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(),0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(),0)
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos

def accuracy(output, target, topk=(1, 5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    logits = torch.rand((10, 1, 256, 256))
    targets = torch.rand((10, 1, 256, 256))
    targets = (targets>0.5).float()

    dice, dice_neg, dice_pos, num_neg, num_pos = dice_metric(logits, targets, threshold=0.5, reduction='none')
    print(f'dice: {dice}, dice_neg: {dice_neg}, dice_pos: {dice_pos}, num_neg: {num_neg}, num_pos: {num_pos}')
