import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings(action='ignore')

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
 
    def forward(self, logits, targets):
        ''' fastai.metrics.dice uses argmax() which is not differentiable, so it 
          can NOT be used in training, however it can be used in prediction.
          see https://github.com/fastai/fastai/blob/master/fastai/metrics.py#L53
        '''
        N = targets.size(0)
        preds = torch.sigmoid(logits)
        #preds = logits.argmax(dim=1) # do NOT use argmax in training, because it is NOT differentiable
        # https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/keras/backend.py#L96
        EPSILON = 1e-7
 
        preds_flat = preds.view(N, -1)
        targets_flat = targets.view(N, -1)
 
        intersection = (preds_flat * targets_flat).sum()#.float()
        union = (preds_flat + targets_flat).sum()#.float()
        
        loss = (2.0 * intersection + EPSILON) / (union + EPSILON)
        loss = 1 - loss / N
        return loss

class Weight_Soft_Dice_Loss(torch.nn.Module):
    def __init__(self, weight=[0.2, 0.8]):
        super(Weight_Soft_Dice_Loss, self).__init__()
        
        self.weight = weight
 
    def forward(self, logits, targets):
        N = targets.size(0)
        preds = torch.sigmoid(logits)
        #preds = logits.argmax(dim=1) # do NOT use argmax in training, because it is NOT differentiable
        # https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/keras/backend.py#L96
        EPSILON = 1e-7
 
        preds_flat = preds.view(N, -1)
        targets_flat = targets.view(N, -1)
        assert(preds_flat.size()==targets_flat.size())
        
        w = targets_flat.detach()
        w = w*(self.weight[1]-self.weight[0])+self.weight[0]
        
        preds_flat = w*(preds_flat*2-1) # convert to [0,1] --> [-1, 1]
        targets_flat = w*(targets_flat*2-1)
 
        intersection = (preds_flat * targets_flat).sum(-1)#.float()
        union = (preds_flat*preds_flat).sum(-1) + (targets_flat*targets_flat).sum(-1)
        
        dice = (2.0 * intersection + EPSILON) / (union + EPSILON)
        loss = (1 - dice)
        
        return loss.mean()

class BCELoss(torch.nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
 
    def forward(self, logits, targets):
        logit = logits.contiguous().view(-1)
        truth = targets.contiguous().view(-1)
        assert(logit.shape==truth.shape)

        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

        return loss.mean()

class Weight_BCELoss(torch.nn.Module):
    def __init__(self, weight_pos=0.25, weight_neg=0.75):
        super(Weight_BCELoss, self).__init__()
        
        self.weight_pos = weight_pos
        self.weight_neg = weight_neg
 
    def forward(self, logits, targets):
        logit = logits.contiguous().view(-1)
        truth = targets.contiguous().view(-1)
        assert(logit.shape==truth.shape)

        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
        if 0:
            loss = loss.mean()
        if 1:
            pos = (truth>0.5).float()
            neg = (truth<0.5).float()
            pos_weight = pos.sum().item() + 1e-12
            neg_weight = neg.sum().item() + 1e-12
            loss = (self.weight_pos*pos*loss/pos_weight + self.weight_neg*neg*loss/neg_weight).sum()

        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()

class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        self.dice = DiceLoss()

    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(self.dice(input, target))
        return loss.mean()

class Lovasz_Loss(torch.nn.Module):
    def __init__(self, margin=[1,5]):
        super(Lovasz_Loss, self).__init__()
        
        self.margin = margin
 
        def compute_lovasz_gradient(truth): #sorted
            truth_sum    = truth.sum()
            intersection = truth_sum - truth.cumsum(0)
            union        = truth_sum + (1 - truth).cumsum(0)
            jaccard      = 1. - intersection / union
            T = len(truth)
            jaccard[1:T] = jaccard[1:T] - jaccard[0:T-1]

            gradient = jaccard
            return gradient

        def lovasz_hinge_one(logit , truth):
            m = truth.detach()
            m = m*(margin[1]-margin[0])+margin[0]

            truth = truth.float()
            sign  = 2. * truth - 1.
            hinge = (m - logit * sign)
            hinge, permutation = torch.sort(hinge, dim=0, descending=True)
            hinge = F.relu(hinge)

            truth = truth[permutation.data]
            gradient = compute_lovasz_gradient(truth)

            loss = torch.dot(hinge, gradient)
            return loss
        self.lovasz_hinge_one = lovasz_hinge_one
        

    def forward(self, logit, truth):
        batch_size = len(logit)
        logit = logit.view(batch_size,-1)
        truth = truth.view(batch_size,-1)
        assert(logit.shape==truth.shape)

        batch_size = len(truth)
        loss = torch.zeros(batch_size).cuda()
        for b in range(batch_size):
            l, t = logit[b].view(-1), truth[b].view(-1)
            loss[b] = self.lovasz_hinge_one(l, t)
        
        return loss.mean()

if __name__ == '__main__':
    logits = torch.rand((4, 1, 256, 256))
    targets = torch.rand((4, 1, 256, 256))
    targets = (targets>0.5).float()
    # print(targets.data)

    # DiceLoss
    diceloss = DiceLoss()(logits, targets)
    print(f'DiceLoss: {diceloss}')

    # Weight_Soft_Dice_Loss
    weight_soft_dice_loss = Weight_Soft_Dice_Loss(weight=[0.2, 0.8])(logits, targets)
    print(f'Weight_Soft_Dice_Loss: {weight_soft_dice_loss}')

    # BCELoss
    bceloss = BCELoss()(logits, targets)
    print(f'BCELoss: {bceloss}')

    # Weight_BCELoss
    weight_bceloss = Weight_BCELoss(weight_pos=0.25, weight_neg=0.75)(logits, targets)
    print(f'Weight_BCELoss: {weight_bceloss}')

    # FocalLoss
    weight_bceloss = FocalLoss(gamma=2)(logits, targets)
    print(f'FocalLoss: {weight_bceloss}')

    # MixedLoss FocalLoss + Dice_Loss
    mixedloss = MixedLoss(alpha=1, gamma=2)(logits, targets)
    print(f'MixedLoss: {mixedloss}')

    # FocalLoss
    lovasz_loss = Lovasz_Loss(margin=[1, 5])(logits, targets)
    print(f'Lovasz_Loss: {lovasz_loss}')
