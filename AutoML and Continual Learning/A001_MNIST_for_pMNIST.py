###===###
# Nic5472K
# A script for applying an MLP for permuted pixel MNIST

###===###
# Defining Dependencies
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
# Seeding
seed = 0
random.seed(                seed)
np.random.seed(             seed)
torch.manual_seed(          seed)
torch.cuda.manual_seed(     seed)

###===###
# Preparing dataset
#---
# batch size
My_BS   = 256

#---
# loaders
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

#---
# image vectorisation will be shown a bit later
# creating 1 permutation mask
My_Perm = [atr for atr in range(784)]
random.shuffle(My_Perm)

###===###
# Defining my base learner
# our base learner will be a 2-layer perceptron with a linear softmax layer
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        # the 2LP part
        self.L1 = nn.Linear(784, 100)
        self.L2 = nn.Linear(100, 100)

        # the softmax part
        self.L3 = nn.Linear(100, 10)

        self.init_weight()

    def init_weight(self):

        for param in self.parameters():
            if len(param.shape) == 1:
                # for bias
                param.data.copy_(
                    torch.zeros_like(param))
            else:
                # for weights
                std = 1.0 * math.sqrt(
                                2.0 / sum( list(param.shape) )
                                )
                a   = math.sqrt(3.0) * std
                param.data.copy_(
                    (torch.rand_like(param) * 2 - 1) * a
                    )
        
    def forward(self, x):

        # the 2LP part
        x1 = torch.relu(self.L1(x))
        x2 = torch.relu(self.L2(x1))

        # the linear softmax classifer part
        x3 = self.L3(x2)
        y_hat = F.log_softmax(x3, dim = 1)

        return y_hat

###===###
# Defining a utility evaluation function for later on
def My_test(model):

    # put the base learner into evaluation mode
    model.eval()
    # a counter for the amount of correct examples
    correct_tot = 0

    # for every pair of labelled data therein the loader
    for data, target in test_loader:
        # vectorisation of the image
        data        = data.view(My_BS, -1).cuda()
        # applying the fixed permutation mask
        data        = data[:, My_Perm]
        # inference
        _, y_hat    = torch.max(
                        model(data).data.cpu(),
                        1, keepdim=False
                        )
        # counting the correctly predicted ones
        correct_tot += (target == y_hat).float().sum()

    # creating an accuracy
    acc = correct_tot / ( len(test_loader) * My_BS)

    return acc

###===###
# Main bit

#---
# Define our base learner
MyModel     = MLP()
MyModel     = MyModel.cuda()

# Define our loss
criterion   = nn.CrossEntropyLoss()

# We will use SGD to update our base learner
optimiser   = optim.SGD(MyModel.parameters(), lr = 0.1)

#---
# misc
print("==" * 20)
print("The training phase has started")

# Let's update our base learner for 5 epochs
MaxEP = 5
print("the MLP will undergo {} epochs of training".format(MaxEP))
    
for epoch in range(MaxEP):

    # for each pair of labelled data
    for batch_id, (data, target) in enumerate(train_loader):

        # vectorise the input data
        data = data.view(My_BS, -1)
        # and apply the permutation mask
        data = data[:, My_Perm]

        # put the pair in GPU
        data, target = data.cuda(), target.cuda()

        # inference
        MyModel.train()
        MyModel.zero_grad()
        y_hat = MyModel(data)

        # compute loss
        loss = criterion(y_hat, target)
        loss.backward()

        # update the learner parameters
        optimiser.step()

    #---
    # misc
    print("--" * 20)
    print("Epoch {} is done".format(epoch + 1))

#---
# end of trianing evaluation
print("<>" * 20)
print("The training has ended and...")
Final_Accuracy = My_test(MyModel)
Final_Accuracy = Final_Accuracy.numpy()
print("the final accuracy is {}%".\
      format(
          round(Final_Accuracy * 100, 2)
          )
      )

        












































