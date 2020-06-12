###===###
import  torch
import  torch.nn            as      nn
import  torch.nn.functional as      F
from    torch.nn.utils      import  weight_norm

###===###
# Differences to the pMNIST code:
#   (1) pMNIST tests under a multiple-to-one setup
#       so it takes the final element of the last TCN layer
#       on the other hand
#       JSB tests under a 1-1 setup
#       so it takes the entire final TCN layer
#
#   (2) pMNIST used a SML (softmax layer)
#       however, JSB uses a simple linear layer
#       with the sigmoid activation function

###===###
class Causal_Conv1d(nn.Module):
    #---
    def __init__(self, ID, HD, KS, DS):
        super(Causal_Conv1d, self).__init__()

        self.CS = (KS - 1) * DS
        self.C  = nn.Conv1d(
                    ID, HD, KS,
                    dilation = DS,
                    padding = self.CS)
    #---
    def forward(self, x):

        z = self.C(x)
        z = z[:, :, :-self.CS].contiguous()

        return z
                
###===###
class TCN_Block(nn.Module):
    #---
    def __init__(self,
                 ID, HD,
                 KS, DS, Drp):
        super(TCN_Block, self).__init__()

        self.ID     = ID
        self.HD     = HD
        self.Drp    = Drp

        self.C1     = Causal_Conv1d(ID, HD, KS, DS)
        self.C2     = Causal_Conv1d(HD, HD, KS, DS)

        if ID != HD:
            self.Sc = nn.Conv1d(ID, HD, 1)

        else:
            self.Sc = None

        self.init_weights()

    #---
    def init_weights(self):

        for my_layer in self.children():

            if isinstance(my_layer, Causal_Conv1d):
                my_layer.C.weight.data.copy_(my_layer.C.weight.data * 0.01)
                my_layer.C.bias.data.copy_(  my_layer.C.bias.data   * 0.00)

            if isinstance(my_layer, nn.Conv1d):
                my_layer.weight.data.copy_(my_layer.weight.data * 0.01)
                my_layer.bias.data.copy_(  my_layer.bias.data   * 0.00)

    #---
    def forward(self, x):

        RC1 = torch.relu(self.C1(x))
        if self.training:
            RC1 = F.dropout(RC1, self.Drp)

        RC2 = torch.relu(self.C2(RC1))
        if self.training:
            RC2 = F.dropout(RC2, self.Drp)

        Stream1 = RC2

        if self.ID != self.HD:
            Stream2 = self.Sc(x)
            
        else:
            Stream2 = x

        Z = torch.relu(Stream1 + Stream2)

        return Z

###===###
class My_TCN(nn.Module):
    #---
    def __init__(self, ID, HD, OD, LS, KS, DS, Drp):
        super(My_TCN, self).__init__()

        TCN_layers = []
        
        All_Ds = [ID] + [HD] * LS

        for D_ind in range(1, len(All_Ds)):

            Cur_ID = All_Ds[D_ind - 1]
            Cur_HD = All_Ds[D_ind]

            Cur_DS = 2 ** (DS - 1)

            TCN_layers += \
                       [TCN_Block(Cur_ID, Cur_HD, KS, Cur_DS, Drp)]

        self.Process = nn.Sequential(*TCN_layers)

        # FC: forward connection
        self.FC = nn.Linear(HD, OD, bias = False)
        self.FC.weight.data.copy_(self.FC.weight.data * 0.01)        

    #---
    def forward(self, x):

        # the original input is of shape
        # torch.Size([XX, 88])
        # where XX is the sequence length
        # the length differs for every sequence
        # so they would need to be processed one after another
        # hence
        # the batch size is 1

        # Since TCN uses Conv1d
        # we will need to ensure that the input dimension is
        # (BS = 1, ID, seq_len)
        # hence
        x = x.transpose(0, 1).unsqueeze(0)

        #self added
        if self.training:
            x = F.dropout(x, p = 0.1)

        # Note (1)
        # there is no longer an x_final here
        x_Processed = self.Process(x)

        # x_Processed is of dimension
        # (1, HD = ID, seq_len)
        # note, the hidden dimension for JSB
        # is same as the input dimension for JSB
        # but the linear layer needs an input with dimension
        # (seq_len, ID)
        # hence
        x_Processed = x_Processed.squeeze(0).transpose(0, 1)
        
        y_FC        = self.FC(x_Processed)

        #self added
        if self.training:
            y_FC += torch.tensor([1e-2]).cuda()
        
        # Note (2)
        # sigmoid function as activation function
        y_hat       = torch.sigmoid(y_FC)       
        
        return y_hat
        


















































































