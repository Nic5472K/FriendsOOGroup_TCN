###===###
import  numpy       as      np
from    scipy.io    import  loadmat

import  torch

###===###
def Prepare_JSB_data():

    data = loadmat('./A000_Data_Related_Stuff/JSB_Chorales.mat')

    X_train = data['traindata'][0]
    X_valid = data['validdata'][0]
    X_test  = data[ 'testdata'][0]
    
    for data in [X_train, X_valid, X_test]:
        for i in range(len(data)):
            data[i] = torch.Tensor(data[i].astype(np.float64))

    return X_train, X_valid, X_test
