###===###
# Nic5472K
# this script was first made use in A003
# the main difference to its Z003 counterpart
# is that we measure all accuracies on all tasks

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
def My_test2(model, All_Perms, test_loader):

    model.eval()
    
    # create an empty dictionary
    ResultsDict = {}
    
    # for each task
    for atr in range(len(All_Perms)):
        # first create the key for the dictionary
        TaskID = 'Task {}'.format(atr + 1)

        # reinitialise all counters for each task    
        correct_tot = 0
        amount_observed = 0

        # remember to apply the appropriate mask
        My_Perm = All_Perms[atr]

        # now the usual stuff
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

        # record the information
        ResultsDict[TaskID] = acc.numpy()

    return ResultsDict
