###===###
# Nic5472K
# Unlike A002,
# here we will be sequentially learning 5 tasks
#   i.e. pMNIST with 5 different masks
# using just one base learner

###===###
import  random
import  math
import  numpy                   as      np
import  matplotlib.pyplot       as      plt
#---
import  torch
import  torch.nn                as      nn
import  torch.optim             as      optim
import  torch.nn.functional     as      F
#---
import  torchvision
import  torchvision.transforms  as      transforms
from    torchvision             import  datasets, transforms

#---
from    Z001_DefineLoaders      import  *
from    Z002_DefineLearner      import  *
# the Evaluation scheme has now changed,
# check Z004 for comments
from    Z004_DefineEval2        import  *

###===###
seed = 0
random.seed(                seed)
np.random.seed(             seed)
torch.manual_seed(          seed)
torch.cuda.manual_seed(     seed)

###===###
My_BS   = 256
train_loader, test_loader = ShowMeTheLoaders(My_BS)

###===###
MaxMasks = 5
All_Perms = []
for atr in range(MaxMasks):
    Cur_Perm = [btr for btr in range(784)]
    random.shuffle(Cur_Perm)
    All_Perms.append(Cur_Perm)

###===###
# here we define just 1 model
MyModel = MLP()
MyModel = MyModel.cuda()
#---
criterion   = nn.CrossEntropyLoss()
optimiser   = optim.SGD(MyModel.parameters(), lr = 0.1)

###===###
print("==" * 20)
print("The training phase has started")
print("Here, we train 1 learners sequentially across 5 tasks")
print("Without ever looking back at it anymore")
MaxEP = 5
print("and each task will undergo {} epochs of training\n".format(MaxEP))

###===###
# testing the network prior to training
print("VV" * 20)
print("Testing the model prior to learning")
PreTrained_Results = My_test2(MyModel, All_Perms, test_loader)

print("=-="*5)
print("Task: \t Nothing yet")
print("|| Target \t || Accuracy")
for atr in range(MaxMasks):
    CurrentID   = list(PreTrained_Results.keys())[atr]
    CurrentACC  = PreTrained_Results[CurrentID]
    print("|| {} \t || {}%".\
          format( CurrentID,
                  round(CurrentACC * 100, 2)
                  )
          )
print("")

###===###
# Here we create storages to track the progress of learning
# We will default them as the PreTrained_Results
# and we will update on the go
Old_Results     = PreTrained_Results
Best_Results    = PreTrained_Results
Straight_After  = []
Straight_Before = []

Initial_Results = PreTrained_Results.copy()

###===###
# Also saving information for plotting
Results4Plotting = []
Results4Plotting.append(
    list(np.array(list(Initial_Results.values())))
    )

#---
for TaskID in range(MaxMasks):
    print("++" * 20)
    print("Selecting a new mask")
    My_Perm     = All_Perms[TaskID]
    print("Okidoki, let's get going")
    print("xx" * 20)
    print("Task {}/{}".format(TaskID + 1, MaxMasks))
    #---
    for epoch in range(MaxEP):
        for batch_id, (data, target) in enumerate(train_loader):
            data = data.view(data.shape[0], -1)
            data = data[:, My_Perm]
            data, target = data.cuda(), target.cuda()

            MyModel.train()
            MyModel.zero_grad()

            y_hat   = MyModel(data)
            loss    = criterion(y_hat, target)
            loss.backward()
            optimiser.step()

        #---
        # misc
        print("--" * 20)
        print("Epoch {} is done".format(epoch + 1))

    #---
    # end of task evaluation (EoTE)
    print("")
    print("<>" * 20)
    print("The training has ended for task {}".format(TaskID + 1))
    EoTE_Accuracy = My_test2(MyModel, All_Perms, test_loader)

    Results4Plotting.append(
        list(np.array(list(EoTE_Accuracy.values())))
        )

    # printing the results
    print("=-="*5)
    p1 = "|| Target \t || Current \t"
    p2 = "|| Last \t || Best \t"
    p3 = "|| Diff2Last \t || Diff2Best"
    print(p1 + p2 + p3)
    print("---"*5)

    for atr in range(MaxMasks):
        CurrentID   = list(EoTE_Accuracy.keys())[atr]
        CurrentACC  = EoTE_Accuracy[CurrentID]

        #---
        # update the accuracy straight after training
        if TaskID == atr:
            Straight_After.append(CurrentACC)

        #---
        # update the accuracy of the following task straight before training
        if (TaskID + 1) == atr:
            Straight_Before.append(CurrentACC)

        #---
        # loading PastAcc and PrevBestAcc
        PastAcc     = Old_Results[CurrentID]
        PrevBesAcc  = Best_Results[CurrentID]

        #---
        # updating the best result and
        # finding the best result so far
        if CurrentACC > PrevBesAcc:
            Best_Results[CurrentID] = CurrentACC
            BestAcc = CurrentACC

        else:
            BestAcc = PrevBesAcc

        #---
        # find the differences between results
        CurMinusLast = CurrentACC - PastAcc
        CurMinusBest = CurrentACC - BestAcc

        #---
        q1 =  "|| {} \t || {}% \t".format(
                                        CurrentID,
                                        round(CurrentACC * 100, 2)
                                       )
        q2 = "|| {}% \t || {}% \t".format(
                                        round(PastAcc    * 100, 2),
                                        round(BestAcc    * 100, 2)
                                       )
        q3a = '{}% '.format(round(CurMinusLast  * 100, 2))
        q3b = '{}% '.format(round(CurMinusBest  * 100, 2))

        #---
        # adding a bit of status labelling between
        # the current result and the most immediate past result
        # if improved                <+>
        # if decreased (-10,    0]%  <.>
        #              (-20,  -10]%  <*>
        #              (-inf, -20]%  <**>
        if      (CurMinusLast  * 100) >  0:
                    q3a = q3a  + '<+>'
        elif -1*(CurMinusLast  * 100) > 20:
                    q3a = q3a  + '<**>'
        elif -1*(CurMinusLast  * 100) > 10:
                    q3a = q3a  + '<*>'
        elif -1*(CurMinusLast  * 100) >  0:
                    q3a = q3a  + '<.>'

        #---
        # now do the same labelling but respective to the best result
        if      (CurMinusBest  * 100) >= 0:
            #---
            # specify if it is zero-shot learning
            if atr > TaskID:
                q3b = q3b  + '<ZSL>'
            # or if it is positive baward's transfer
            else:
                q3b = q3b  + '<+>'
                
        elif -1*(CurMinusBest  * 100) > 20:
                    q3b = q3b  + '<**>'
        elif -1*(CurMinusBest  * 100) > 10:
                    q3b = q3b  + '<*>'
        elif -1*(CurMinusBest  * 100) >  0:
                    q3b = q3b  + '<.>'        

        #---
        q3 = "|| {} \t || {}".format(q3a, q3b)

        #---
        print(q1 + q2 + q3)

    #---
    # updating the old results
    Old_Results = EoTE_Accuracy
        
    print("")

#---
# Here we will print all matrics
# including:
#   Benchmark: Oracle method
#       The average accuracy of all best results
#---
#   (1) ACC: Average accuracy
#               After entire sequential training
#   (2) BWT: Backwards transfer
#               Average accuracy between final and straight after training
#   (3) FWT: Forwards  transfer
#               Average accuracy between initial and straight before training

print("^^" * 20)
print("Final Summary:")
print("---" * 5)

#---
Oracle  = np.mean(list(Best_Results.values()))
print("Oracle:  {}%".\
      format( round( Oracle * 100, 2) )
      )

#---
ACC     = np.mean(list(EoTE_Accuracy.values()))
print("ACC: \t {}%".\
      format( round( ACC * 100, 2) )
      )

#---
BWT     = np.array(list(EoTE_Accuracy.values())) - \
          np.array(Straight_After)
BWT = BWT[:-1]
BWT = BWT / (MaxMasks - 1)
BWT = np.mean(BWT)
print("BWT: \t {}%".\
      format( round( BWT * 100, 2) )
      )

#---
FWT     = np.array(Straight_Before) - \
          np.array(list(Initial_Results.values()))[1:]
FWT = np.mean(FWT)
print("FWT: \t {}%".\
      format( round( FWT * 100, 2) )
      )

###===###
# plotting the process
if 1:
    Results4Plotting = np.array(Results4Plotting)
    xVal = [5 * (i+1) for i in range(MaxMasks)]
    xVal = [0] + xVal

    Legend_Names = []
    
    for atr in range(MaxMasks):
        Cur_Y = Results4Plotting[:, atr]
        plt.plot(xVal, Cur_Y)

        Legend_Names += ['Task {}'.format(atr)]

    for atr in range(len(xVal)):
        xLoc = xVal[atr]
        plt.axvline(x = xLoc, linestyle='--', color = 'k')

    plt.legend(Legend_Names, loc = 'upper left')
    plt.ylim([-0.1, 1.1])

    plt.title( 'Progress of Task-wise Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    plt.show()











    
