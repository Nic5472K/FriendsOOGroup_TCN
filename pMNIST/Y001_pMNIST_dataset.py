###===###
import  torch

import  torchvision
import  torchvision.transforms  as      transforms
from    torchvision             import  datasets, transforms

###===###
def load_data(BS):
    kwargs = {'pin_memory': True}

    train_loader = \
        torch.utils.data.DataLoader(
            datasets.MNIST(
                './A000_Data_Related_Stuff',
                train = True, download = True,
                transform = \
                    transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size = BS,
            shuffle = True,
            **kwargs)     

    test_loader = \
        torch.utils.data.DataLoader(
            datasets.MNIST(
                './A000_Data_Related_Stuff',
                train = False, download = True,
                transform = \
                    transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size = BS,
            shuffle = True,
            **kwargs)

    return train_loader, test_loader
