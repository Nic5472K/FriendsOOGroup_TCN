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
def ShowMeTheLoaders(My_BS):
    kwargs = {'pin_memory': True}

    train_loader = \
        torch.utils.data.DataLoader(
            datasets.MNIST(
                './data_M',
                train = True, download = True,
                transform = \
                    transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size = My_BS,
            shuffle = True,
            **kwargs)     

    test_loader = \
        torch.utils.data.DataLoader(
            datasets.MNIST(
                './data_M',
                train = False, download = True,
                transform = \
                    transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size = My_BS,
            shuffle = True,
            **kwargs)

    return (train_loader, test_loader)
