###===###
# Nic5472K

###===###
import  random
import  math
import  numpy                   as      np
#---
import  torch
import  torch.nn                as      nn
import  torch.optim             as      optim
import  torch.nn.functional     as      F
#---
import  torchvision
import  torchvision.transforms  as      transforms
from    torchvision             import  datasets, transforms

###===###
def My_test(model, My_Perm, test_loader):

    model.eval()
    correct_tot = 0
    amount_observed = 0

    for data, target in test_loader:
        data        = data.view(data.shape[0], -1).cuda()
        data        = data[:, My_Perm]
        _, y_hat    = torch.max(
                        model(data).data.cpu(),
                        1, keepdim=False
                        )
        correct_tot += (target == y_hat).float().sum()
        amount_observed += data.shape[0]

    acc = correct_tot / amount_observed

    return acc
