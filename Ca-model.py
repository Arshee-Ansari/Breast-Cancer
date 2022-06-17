import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

# loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

# adding the 'target' column to the data frame
data_frame['label'] = breast_cancer_dataset.target

# 1 --> Benign
# 0 --> Malignant
sns.countplot(data_frame['label'],label="count")

#Separating the features and target
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

#Splitting the data into training data & Testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

def models(X_train,Y_train):
        # logistic regression
        from sklearn.linear_model import LogisticRegression
        log=LogisticRegression(random_state=0)
        log.fit(X_train,Y_train)
        
        
        # Decision Tree
        from sklearn.tree import DecisionTreeClassifier
        tree=DecisionTreeClassifier(random_state=0,criterion="entropy")
        tree.fit(X_train,Y_train)
        
        # Random Forest
        from sklearn.ensemble import RandomForestClassifier
        forest=RandomForestClassifier(random_state=0,criterion="entropy",n_estimators=10)
        forest.fit(X_train,Y_train)
        
        # Support Vector Machine
        from sklearn.svm import SVC
        svm = SVC(kernel='linear', random_state=0, C=0.1)
        svm.fit(X_train,Y_train)
        
        # K-Nearest Neighbor
        from sklearn.neighbors import KNeighborsClassifier
        nbrs = KNeighborsClassifier(n_neighbors=10)
        nbrs.fit(X_train,Y_train)
        
        # Print the models accuracy on the training data 
        print('[0]logistic regression accuracy:',log.score(X_train,Y_train))
        print('[1]Decision tree accuracy:',tree.score(X_train,Y_train))
        print('[2]Random forest accuracy:',forest.score(X_train,Y_train))
        print('[3]Support Vector Machine accuracy:',svm.score(X_train,Y_train))
        print('[4]K-Nearest Neighbor accuracy:',nbrs.score(X_train,Y_train))
        
        return log,tree,forest,svm,nbrs

# Accuracy on train data
model=models(X_train,Y_train)

#Accuracy on test data
for i in range(len(model)):
    X_test_prediction = model[i].predict(X_test)
    test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
    print("Model",i)
    print('Accuracy on test data = ', test_data_accuracy)

Models=['Logistic Regression ','Decision Tree','Random Forest','Support Vector Machine','K-Nearest Neighbor']
Accuracy=[92,90,93,94,93]

xpos = np.arange(len(Models))
xpos

# prediction of Support Vector Machine
input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model[3].predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Breast cancer is Malignant')

else:
  print('The Breast Cancer is Benign')


filename="Cancer_B"
pickle.dump(model[3], open(filename,'wb'))

loaded_model = pickle.load(open(filename,'rb'))
loaded_model.predict(X_test)