def svmComp():
    # SVM(Support Vector Machine)
    # Classification template
    # Classification is based on the probability, so probability can't be less
    # ----- than zero and thus regression is not applicable here
    # ----- Logistics regression is used  to predict the probabilities
    # Importing the libraries
    import numpy as np 
    import matplotlib.pyplot as plt
    import pandas as pd

    #Importing the dataset
    dataComp1 = pd.read_csv('f:/django/GOOG.csv')
    dataComp2 = pd.read_csv('f:/django/IBM.csv')
    dataComp3 = pd.read_csv('f:/django/AAPL.csv')
    y = pd.read_csv('f:/django/fifteen.csv')
    #y_test = pd.read_csv('f:/django/y_test.csv', sep='delimiter', header=None)
    y_test_list=[0,1,0]
    #y_test_frame = pd.DataFrame(y_test_list, index =['0','1', '2'], 
    #                                              columns =['Profits']) 
    y_test = np.asarray(y_test_list)
    #Making of test set
    Xt=[1]
    X1_test = dataComp1.iloc[[1],[6]]
    X1_test['Companies']=Xt

    Xt=[2]
    X2_test = dataComp2.iloc[[1],[6]]
    X2_test['Companies']=Xt

    Xt=[3]
    X3_test = dataComp3.iloc[[1],[6]]
    X3_test['Companies']=Xt

    frames=[X1_test,X2_test,X3_test]

    X_test = pd.concat(frames)      #X-test set ready


    y_list = y.values.tolist()
    l=len(y_list)
    y_List=[]
    for j in range(0,l-1):
        y_List.append(y_list[j][0])
    y_List.append(y_list[l-1][0])
    #y_train= np.asarray(y_List)

    n1 = len(dataComp1)
    X1 = dataComp1.iloc[[n1-5,n1-4,n1-3,n1-2,n1-1],[3,4,6]]  # a:b shows a to b-1

    n2 = len(dataComp2)
    X2 = dataComp2.iloc[[n1-5,n1-4,n1-3,n1-2,n1-1],[3,4,6]]

    n3 = len(dataComp3)
    X3 = dataComp3.iloc[[n1-5,n1-4,n1-3,n1-2,n1-1],[3,4,6]]

    frames = [X1,X2,X3]

    X_all = pd.concat(frames)

    X_list = X_all.values.tolist()
    i=0
    profit=[]
    l=len(X_list)

    for i in range(0,l):
        #print("tsettttt......")
        if X_list[i][0]-X_list[i][1]<0:
            profit.append(0)
        elif X_list[i][0]-X_list[i][1]>0:
            profit.append(1)

    if X_list[l-1][0]-X_list[l-1][1]<0:
        #print(X_list[l-1][0],X_list[l-1][1])
        profit.append(0)
    elif X_list[l-1][0]-X_list[l-1][1]>0:
        #print(X_list[l-1][0],X_list[l-1][1])
        profit.append(1)
    # profit is the main training set
    #X_all['Profits']=profit

    mainSeries = X_all.iloc[:,2]

    X_train = mainSeries.to_frame()    
    X_train['Companies']=y_List
    y_train=np.asarray(profit)

    #it must relate to the renaming and depreaction of cross_validation 
    #------submodule to model_selection.
    #------ Try substituting cross_validation -> model_selection
    #Splitting the dataset into the training set and Test set
    #from sklearn.model_selection import train_test_split
    #X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

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

    #Visualizing the test set results
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




    """
    #Visualizing the training set results
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_test, y_train
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
    """
    # Red means -> user DON'T buys car &  Green means -> user  buys car
    # Each point in the graph is a user in a social network with a salary and age
    # eg. age is high, salary is low but still a person buys a car


