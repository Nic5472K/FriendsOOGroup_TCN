###===###
import torch
import torch.nn.functional  as F

import numpy                as np
import random

###===###
# Note:
#   The loss function of JSB is quite interesting
#   and it is as defined below
#---
# it is
# (1)   the negative of
# (2)   the trace of
# (3)   the sum of 2 similarity matrices
#---
# Why?
# (1)   ensures that we have a minimisation target
# (2)   describes what the location of minimisation target
# (3)   the product of matrix A and the transpose of matrix B
#       if A and B are identical
#       then we a symmetric matrix
#       else it won't be
#       try
#       ###===###
#       >>> tim
#       tensor([[1., 1., 1.],
#               [2., 2., 2.]])
#       >>> torch.mm(tim, tim.t())
#       tensor([[ 3.,  6.],
#               [ 6., 12.]])
#       >>> torch.mm(tim, torch.randn_like(tim).t())
#       tensor([[-1.2864,  1.7597],
#               [-2.5728,  3.5194]])
#
#       ###===###
#       the trace of such a matrix product incorporates
#       noise from the uncertainty of the network
#---
# there is also an important (4)th element
# (4)   the network uses log functions to heavily penalise network uncertainty

def MatrixSymmetryLoss(y_hat, y):

    My_loss = (
        # feature (1)
            -1 *
        # feature (2)
            torch.trace(
        # feature (3) part 1
                torch.matmul(
                    y,
        # feature (4)
                    torch.log(y_hat).t()) +
        # feature (3) part 2
                torch.matmul(
                    (1 - y),
        # feature (4) again                
                    torch.log(1- y_hat).t())
                ))
    
    return My_loss

###===###
def train(train_loader, optimiser, model, cur_ep):

    model.train()

    TLPP = 0
    SLen = 0        # samples length per print

    random.shuffle(train_loader)
    
    for seq_id, every_sequence in enumerate(train_loader):

        # each sequence is different in length
        x_len = every_sequence.shape[0] - 1

        # the aim is to predict the forth coming sequence
        # so if we have
        # inputs from instances 1 to 10
        # our
        # outputs are from instances 2 to 11
        # similarly if given 6~21,
        # then predict for 7~22
        x = every_sequence[:-1]
        y = every_sequence[1:]
        
        x, y = x.cuda(), y.cuda()

        optimiser.zero_grad()
        y_hat = model(x)
        loss = MatrixSymmetryLoss(y_hat, y)

        # applies gradient clipping to present
        # log(y_hat) from exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.4)
        
        loss.backward()

        optimiser.step()

        TLPP += loss
        SLen += x_len

    if np.mod(cur_ep, 5) == 0:
        print('-' * 20)
        print('Epoch: \t\t {}'.\
                  format(cur_ep))
        print('Average Loss: \t {}'.\
                  format(round(TLPP.detach().cpu().numpy() / SLen, 5)))

###===###
def test(test_loader, model, cur_ep, Loss_Tracker = None):

    model.eval()

    total_loss  = 0
    total_len   = 0

    with torch.no_grad():

        for every_sequence in test_loader:

            x_len = every_sequence.shape[0] - 1

            x = every_sequence[:-1]
            y = every_sequence[1:]

            x, y = x.cuda(), y.cuda()

            y_hat   = model(x)
            loss = MatrixSymmetryLoss(y_hat, y)

            total_loss  += loss.detach().cpu().numpy()
            total_len   += x_len

        if cur_ep == 9999:
            # it is over 9000!!!
            print('-' * 20)
            print('This pre-trained model has')
            print('Average Loss: \t {}'.\
                      format(round(total_loss / total_len, 5)))

        elif np.mod(cur_ep, 5) == 0:
            print('=' * 20)
            print('Testing has finished')
            print('Average Loss: \t {}'.\
                      format(round(total_loss / total_len, 5)
                              ))

        if not Loss_Tracker:
            Loss_Tracker = []
        
        Loss_Tracker.append(total_loss / total_len)

    return Loss_Tracker










            
