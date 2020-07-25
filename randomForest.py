import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

### Load data as Pandas DataFrame
df = pd.read_csv('./mushrooms.csv')

### Here we shuffle the data frame over its all rows
df = df.sample(frac=1).reset_index(drop=True)


### preprocessed labels 
### ...the data set contains only alphanumeric values which we need to 
### transform into numerical ones
Y = np.asarray([0 if val == 'e' else 1 for val in df['class']])


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
### Because decision trees needs very less training data in comparson with artificial 
### neural networks, we use only 20 % of our data as training data and 80 % for 
### testing the model.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8)

 
### Because some of the data has include non defined attribut values which we 
### have transformed to np.nan, we need to define a strategy to deal with that.
### The SimpleInputer replaces nan values by the mean value of all values
### NOTICE: However, we should ask ourselves whether this strategy is the best 
### in this context. After all, it's all about poisonous mushrooms...

imp       = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_train = imp.fit(X_train)      


### Now, we are going to train the model
X_train = imp.transform(X_train)
clf     = RandomForestClassifier(max_depth=7, n_estimators=5, random_state=0)
clf     = clf.fit(X_train, Y_train)                     


### Here we prepare the model test
imp      = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_test = imp.fit(X_test)
X_test   = imp.transform(X_test)


### model data for test the model 
test_output = clf.predict(X_test)


### As last step we want to compute the accuracy of the model      
accuracy = accuracy_score(test_output, Y_test)      

print('The accuracy of this model is: ',accuracy)













