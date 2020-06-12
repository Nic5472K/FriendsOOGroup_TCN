###===###
#---
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
from    Y011_JSB_dataset        import  *
from    Y012_JSB_model          import  *
from    Y013_JSB_TrainTest      import  *

###===###
seed = 1
random.seed(                seed)
np.random.seed(             seed)
torch.manual_seed(          seed)
torch.cuda.manual_seed_all( seed)

###===###
#---
My_EP       = 100

#---
# The fully parametric setting is documented in
# page 12 of the original paper
My_ksize    = 3
My_dsize    = 2
My_ID       = 88            # for the 88 notes on a piano
My_HD       = 150
My_OD       = 88            # for predicting the outputting 88 notes
My_Drp      = 0.5
My_layers   = 2

#---
My_LR       = 1e-3

###===###
# loading the data
# dimensionalities:
#   train_loader        (299, )
#   valid_loader        (76,  ) <- optional
#   test_loader         (77,  )
#---
train_loader, _, test_loader = Prepare_JSB_data()

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

Loss_Tracker = None

# for testing a pre-trained model
Loss_Tracker = test(test_loader, BaseLearner, 9999, Loss_Tracker)

# usual training
for current_epoch in range(My_EP):
    train(train_loader, optimiser, BaseLearner, current_epoch + 1)
    
    Loss_Tracker = \
    test( test_loader,             BaseLearner, current_epoch + 1,
          Loss_Tracker)

    # fine-tune the learning rate
    if np.mod(current_epoch +1, 10) == 0:
        if Loss_Tracker[-1] > Loss_Tracker[-3]:
            optimiser.param_groups[0]['lr'] /= 10.

            My_Params = torch.load('Z011_SD')
            BaseLearner.load_state_dict(My_Params)

        else:
            # save a copy of the best model yet
            My_Params = BaseLearner.state_dict()
            torch.save(My_Params, 'Z011_SD')

print('=*' * 10)
print('Training has terminated')











