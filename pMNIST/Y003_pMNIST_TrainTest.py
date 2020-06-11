###===###
import torch
import torch.nn.functional  as F

import numpy                as np

###===###
def train(train_loader, P_mask, optimiser, model, cur_ep):

    model.train()

    TLPP = 0        # total loss per print
    SPP  = 0        # samples per print
    
    for batch_id, (x, y) in enumerate(train_loader):

        x_BS = x.shape[0]

        x = x.view(x_BS, 1, -1)
        if P_mask:
            x = x[:, :, P_mask]
        
        x, y = x.cuda(), y.cuda()

        optimiser.zero_grad()
        y_hat = model(x)
        loss = F.nll_loss(y_hat, y)
        loss.backward()

        optimiser.step()

        TLPP += loss
        SPP  += x_BS

        if np.mod( (batch_id + 1), int(len(train_loader) / 5) ) == 0:
            print('-' * 20)
            print('Epoch: \t\t {}'.\
                      format(cur_ep))
            print('Processed: \t {}%'.\
                      format(
                          round((batch_id + 1)/ len(train_loader) * 100, 2)
                          ))
            print('Average Loss: \t {}'.\
                      format(round(TLPP.detach().cpu().numpy() / SPP, 8)))

            TLPP = 0
            SPP  = 0

###===###
def test(test_loader, P_mask, model):

    model.eval()

    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for x, y in test_loader:

            x_BS = x.shape[0]

            x = x.view(x_BS, 1, -1)
            if P_mask:
                x = x[:, :, P_mask]

            x, y = x.cuda(), y.cuda()

            y_hat   = model(x)
            pred    = y_hat.data.max(1, keepdim = True)[1]

            total_correct += pred.eq(y.data.view_as(pred)).cpu().sum().numpy()
            total_samples += x_BS

        print('=' * 20)
        print('Testing has finished')
        print('Accuracy: \t\t {}%'.\
                  format( round(total_correct / total_samples * 100, 2)
                          ))

    










            
