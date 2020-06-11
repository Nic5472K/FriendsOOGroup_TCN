###===###
import  torch
import  torch.nn            as      nn
import  torch.nn.functional as      F
from    torch.nn.utils      import  weight_norm

###===###
# Reminder:
# My_TCN        requires
# TCN_Block,    which itself requires
# Causal_Conv1d

###===###
class Causal_Conv1d(nn.Module):
    ## see details of Causal_Conv1d in Doc001 and Doc002
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

        self.SML = nn.Linear(HD, OD, bias = False)
        self.SML.weight.data.copy_(self.SML.weight.data * 0.01)        

    #---
    def forward(self, x):

        x_Processed = self.Process(x)
        x_final     = x_Processed[:, :, -1]

        y_SM        = self.SML(x_final)
        y_hat       = F.log_softmax(y_SM, dim = 1)
        
        return y_hat
        


















































































