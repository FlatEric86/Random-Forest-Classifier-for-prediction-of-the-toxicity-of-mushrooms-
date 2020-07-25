import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt


### Here we define that the model is computed at the cpu
### In this case we use the cpu instead of gpu because of overhead
device   = torch.device('cpu')



### Load data as Pandas DataFrame
df = pd.read_csv('./mushrooms.csv')

### Here we shuffle the data frame over its all rows
df = df.sample(frac=1).reset_index(drop=True)


### preprocessed labels 
### ...the data set contains only alphanumeric values which we need to 
### transform into numerical ones
### I that case, we transform the binary alphanumrical label into
### a vector with to elements which are either one or zero
### [1, 0] means poisonous and [0, 1] eatable
Y = np.asarray([[0, 1] if val == 'e' else [1, 0] for val in df['class']])



### To transform the attribute values of the other features, we use a map
### which map each lowercase letter to a number in ascending wise
MAP = {}
for letter in range(97,123):
    MAP.update({chr(letter):letter-97})


### Here we transform the attributes 
X = []    
for i in tqdm(range(len(df))):

    ### Some of attribute values are not defined and declared as "?"
    ### That's why we transform these to NAN that can be handled by scikit learn
    vec = [MAP[val] if val != '?' else np.nan for val in df.iloc[i][1:]]   
     
    X.append(vec)
        

### Here we define training data and test data.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8)

 
### Because some of the data has include non defined attribut values which we 
### have transformed to np.nan, we need to define a strategy to deal with that.
### The SimpleInputer replaces nan values by the mean value of all values
### NOTICE: However, we should ask ourselves whether this strategy is the best 
### in this context. After all, it's all about poisonous mushrooms...

imp       = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_train = imp.fit(X_train)      
X_train   = torch.tensor(imp.transform(X_train)).float().to(device)
Y_train   = torch.tensor(Y_train).float().to(device)


imp      = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_test = imp.fit(X_test)
X_test   = torch.tensor(imp.transform(X_test)).float().to(device)
Y_test   = torch.tensor(Y_test).float().to(device)



### Here we define our neural network 
class model(nn.Module):

    def __init__(self):
        super(model, self).__init__()
        
        n = 22
        
        self.lin1  = nn.Linear(n, n)
        self.lin2  = nn.Linear(n, n)
        self.lin3  = nn.Linear(n, 2)
       
    def forward(self, x):
    
        x = self.lin1(x)

        x = torch.nn.functional.sigmoid(self.lin2(x))
        x = torch.nn.functional.sigmoid(self.lin3(x))
        
        return x


### With this function we can compute the accuracity of the model at a specific
### epoch
def test_model():
    '''
    Function to cumpute the model accuracy 
    '''
    model_state_i = net(X_test).to(device)

    count = 0
    with torch.no_grad():
        for i in range(len(X_test)):
          
            real_class      = torch.argmax(Y_test[i]).to(device)
            predicted_class = torch.argmax(model_state_i[i]).to(device)
            if predicted_class == real_class:
                count += 1
                           
    return round(count/len(X_test), 3)





### number of training iterations
N_epoch = 1500
### learning rate
lr      = 0.01  
      
### criterion of loss function
criterion = nn.MSELoss()  

### Define a net object and assign the device    
net      = model().to(device)
    
         
### optimizer = torch.optim.SGD(netz.parameters(), lr=lr)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


loss_function = nn.MSELoss()   
        
### Array to store the loss value per training epoch             
LOSS     = []  

### DataFrame to store the model accuracy and the corresponding training
### epoch at which the model was trained      
ACCURACY = pd.DataFrame(columns=['epoch', 'accuracy'])  
    

### TRAINING PROCESS    
for i in range(N_epoch):

    ### set gradient as 0 
    net.zero_grad()
       
    outputs     = net(X_train).to(device)   
    loss        = loss_function(outputs, Y_train)
    
    ### here we save the value of accuracy and the epoch at each 10'th epoch
    ### in the DataFrame ACCURACY
    if i % 10 == 0:
        acc = test_model()
        ACCURACY = ACCURACY.append({'epoch':i, 'accuracy':acc}, ignore_index=True)

    print(80*'~')
    print('EPOCH:',i)
    print(float(loss))
        
    loss.backward()
    optimizer.step()
    
    LOSS.append(float(loss))

  

### Plotsection to visualize the loss behavior over all epochs as well as
### the evulution of the accuracy over the epochs

fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].plot([i for i in range(len(LOSS))], LOSS, color='green', label='loss')
ax[0].set_title('Evolution of Loss over all Epochs')
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Epochs')
ax[0].legend()

 
ax[1].plot                                                                    \
    (                                                                         \
    ACCURACY['epoch'],                                                        \
    ACCURACY['accuracy'],                                                     \
    color='blue', label='accuracy'                                            \
    )                                                                         \
    
ax[1].set_title('Evolution of Accuracy over all Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].legend()

plt.show()
plt.close()



print('The accuracy of this model is: ',ACCURACY['accuracy'].iloc[-1])













