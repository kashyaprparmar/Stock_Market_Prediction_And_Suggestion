def comp_three(comp1,comp2,comp3,year,month,date):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pandas_datareader import \
        data as web  # Package and modules for importing data; this code may change depending on pandas version
    import datetime

    y=int(year)
    m=int(month)
    d=int(date)
    start = datetime.datetime(y, m, d)  # YY,MM,DD
    end = datetime.date.today()

    comp1Str=str(comp1)
    comp2Str=str(comp2)
    comp3Str=str(comp3)
    
    c1 = web.DataReader(comp1Str, "yahoo", start, end)
    c2 = web.DataReader(comp2Str, "yahoo", start, end)
    c3 = web.DataReader(comp3Str, "yahoo", start, end)

    # Below I create a DataFrame consisting of the adjusted closing price of these stocks, first by making a list of these objects and using the join method
    stocks = pd.DataFrame({comp1Str: c1["Adj Close"],
                           comp2Str: c2["Adj Close"],
                           comp3Str: c3["Adj Close"]})

    print(stocks.head())
    stocks.plot(grid = True)
    plt.show()
