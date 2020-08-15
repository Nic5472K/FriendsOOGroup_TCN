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
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.L1 = nn.Linear(784, 100)
        self.L2 = nn.Linear(100, 100)
        self.L3 = nn.Linear(100, 10)

        self.init_weight()

    def init_weight(self):

        for param in self.parameters():
            if len(param.shape) == 1:
                param.data.copy_(
                    torch.zeros_like(param))
            else:
                std = 1.0 * math.sqrt(
                                2.0 / sum( list(param.shape) )
                                )
                a   = math.sqrt(3.0) * std
                param.data.copy_(
                    (torch.rand_like(param) * 2 - 1) * a
                    )
        
    def forward(self, x):

        x1 = torch.relu(self.L1(x))
        x2 = torch.relu(self.L2(x1))
        x3 = self.L3(x2)
        y_hat = F.log_softmax(x3, dim = 1)

        return y_hat
    
