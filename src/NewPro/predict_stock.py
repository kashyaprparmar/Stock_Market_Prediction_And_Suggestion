def pred_value(pre,comp):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    dataset = pd.read_csv('f:/django/aapl_2_data.csv')
    X = dataset.iloc[:, 0].values
    y = dataset.iloc[:,6].values

    X_list = np.array(X).tolist()
    i=0
    while i<len(X_list):
        X_list[i] = str(X_list[i])
        i = i+1
    i=0
    while i<len(X_list):
        X_list[i] = ''.join(e for e in X_list[i] if e.isalnum())
        i=i+1
    i=0
    while i<len(X_list):
        X_l = int(X_list[i])
        X_l = X_l / 100000.0
        X_list[i] = X_l
        i = i+1
    X = np.array(X_list)

    new_len = len(X_list)

    i=0
    while i<new_len:
        X_list[i]=i+1;
        i = i+1
    p = float(pre)
    m = max(X_list) + p
    X = np.array(X_list)




    """
    y_list = np.array(y).tolist()
    X_list = np.array(X).tolist()
    i=0
    while i<p:
        m = m+1
        f = lin_reg_2.predict(poly_reg.fit_transform(m))
        X_list.append(m)
        y_list.append(f)
        i = i+1
    """
    X = X.reshape(-1,1)


    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)

    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree=6)
    X_poly = poly_reg.fit_transform(X)
    poly_reg.fit(X_poly, y)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y)


    lin_reg = LinearRegression()
    lin_reg.fit(X, y)

    poly_reg = PolynomialFeatures(degree=6)
    X_poly = poly_reg.fit_transform(X)
    poly_reg.fit(X_poly, y)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y)

    """
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    
    
    plt.scatter(X, y, color='red')
    plt.plot(X, lin_reg.predict(X), color='blue')
    plt.title('Date vs Stock (Linear Regression)')
    plt.xlabel('Date')
    plt.ylabel('Salary')
    plt.show()
    """

    # Visualising the Polynomial Regression results
    plt.scatter(X, y, color = 'red')
    plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
    plt.title('Stock Market Prediction of ' + comp +  " by Yahoo Finance")
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.text(.96, .94, "Predicted Value={}".format(str(lin_reg_2.predict(poly_reg.fit_transform(m)))), bbox={'facecolor': 'w', 'pad': 5},
             ha="right", va="top", transform=plt.gca().transAxes)
    plt.show()