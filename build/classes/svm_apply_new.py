import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
compStr=['GOOG','IBM','AAPL']
frames1=[]
frames2=[]
y_test_list2=[]
y_train_list2=[]
j=0

for j in range(0,3):
    dataComp = pd.read_csv('f:/django/'+compStr[j]+'.csv')
    
    y_train_frame = dataComp.iloc[100:400,[3,4]]
    y_train_list  = y_train_frame.values.tolist()
    l=300
    for i in range(0,l):
        if y_train_list[i][0]-y_train_list[i][1]<0:
            y_train_list2.append(0)
        elif y_train_list[i][0]-y_train_list[i][1]>0:
            y_train_list2.append(1)
    
    y_test_frame = dataComp.iloc[0:100,[3,4]]
    y_test_list  = y_test_frame.values.tolist()  
    l=100
    for i in range(0,l):
        if y_test_list[i][0]-y_test_list[i][1]<0:
            y_test_list2.append(0)
        elif y_test_list[i][0]-y_test_list[i][1]>0:
            y_test_list2.append(1)
    if y_test_list[l-1][0]-y_test_list[l-1][1]<0:
         y_test_list2.append(0)
    elif y_test_list[l-1][0]-y_test_list[l-1][1]>0:
         y_test_list2.append(1)
    
    X_inter_test =  dataComp.iloc[0:100,[6]]
    Xt=[j+1]*len(X_inter_test)
    
    X_inter_test['Companies']=Xt
    frames1.append(X_inter_test)
    
    X_train_frame = dataComp.iloc[101:401,[5]]
    frames2.append(X_train_frame)
    
     
X_train=pd.concat(frames2)
X_train_list=[1]*300+[2]*300+[3]*300
X_train['Companies']=X_train_list    
y_train=np.asarray(y_train_list2)
X_test = pd.concat(frames1)              #X_test ready
y_test = np.asarray(y_test_list2)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Fitting Classifier to the Training set
#Create your classifier here
from sklearn.svm import SVC
classifier = SVC(kernel='poly',degree=5,random_state=0)
#classifier = SVC(kernel='linear',random_state=0)
classifier.fit(X_train,y_train)
# POLYNOMIAL KERNEL here

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the confusion matrix to check the 
#----- performance of this logistic regression
#----- model
from sklearn.metrics import confusion_matrix
cn = confusion_matrix(y_test,y_pred)
#   8+3=11 wroing predictions   &   65+24 = correct predictions

#Visualising test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Adj Close')
plt.ylabel('Company')
plt.legend()
plt.show()


from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM(Training set)')
plt.xlabel('Adj Close')
plt.ylabel('Company')
plt.legend()
plt.show()
