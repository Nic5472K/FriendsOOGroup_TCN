###===###
# depnedencies
#---
# the usual
import  math
import  random
import  numpy                   as      np
import  matplotlib.pyplot       as      plt

import  torch
import  torch.nn                as      nn
import  torch.nn.functional     as      F
import  torch.optim             as      optim
from    torch.autograd          import  Variable

#---
# specific to this script
from    Y001_pMNIST_dataset     import  *
from    Y002_pMNIST_model       import  *
from    Y003_pMNIST_TrainTest   import  *



###===###
# seeding
seed = 0
random.seed(                seed)
np.random.seed(             seed)
torch.manual_seed(          seed)
torch.cuda.manual_seed_all( seed)

###===###
# hyperparameters
#---
# setup-related
My_BS       = 64            # batch size
My_EP       = 5             # maximum amount of epochs
My_Perm     = False         # rather or not to permute the sequence

#---
# model-related
My_ksize    = 7             # kernel size
My_dsize    = 2             # incremental dilation size
My_ID       = 1             # input dimension
My_HD       = 25            # hidden dimension
My_OD       = 10            # output dimension
My_Drp      = 0.05          # dropout rate

# amount of layers
My_layers = 1
My_RF = 1 + (My_ksize - 1) * (My_dsize ** (My_layers-1))

while My_RF < 784:
    My_layers   += 1
    My_RF       += (My_ksize - 1) * (My_dsize ** (My_layers-1))

#---
# optimiser-related
My_LR       = 2e-3          # learning rate

#---
# additional stuff
if My_Perm:                 # permutation mask
    My_Pmask = torch.Tensor(
                np.random.permutation(784).astype(np.float64)
                ).long()
else:
    My_Pmask = None

###===###
# setup
#---
# dataset
train_loader, test_loader = load_data(My_BS)

#---
# model
BaseLearner = My_TCN(My_ID, My_HD, My_OD,
                     My_layers,
                     My_ksize, My_dsize,
                     My_Drp)
BaseLearner = BaseLearner.cuda()

# printing some stats regarding the model
pp = 0
for p in list(BaseLearner.parameters()):
    nn = 1
    for s in list(p.size()):
        nn = nn * s
    pp += nn

print('<'*10 + '>'*10)

print('For this experiment, our TCN has')
print('KS: \t\t {}'.format(My_ksize))
print('DS: \t\t {}'.format(My_dsize))
print('layers: \t {}'.format(My_layers))
print('# total param: \t {}'.format(pp))

print('<'*10 + '>'*10)

#---
# optimiser
optimiser = optim.Adam(BaseLearner.parameters(), lr = My_LR)

###===###
# Training and testing
print('Training has started')
print('=*' * 10)

for current_epoch in range(My_EP):
    train(train_loader, My_Pmask, optimiser, BaseLearner, current_epoch + 1)
    test(test_loader, My_Pmask, BaseLearner)

print('=*' * 10)
print('Training has terminated')










