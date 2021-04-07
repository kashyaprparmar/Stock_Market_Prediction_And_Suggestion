def onClick3(comp,year,month,date,graph):
    import matplotlib.pyplot as plt  # Import matplotlib
    # This line is necessary for the plot to appear in a Jupyter notebook
    # % matplotlib
    # inline

    year_str = int(year)
    month_str = int(month)
    date_str = int(date)

    comp_str = comp
    graph_type = graph

    import numpy as np
    import pandas as pd
    from pandas_datareader import \
        data as web  # Package and modules for importing data; this code may change depending on pandas version
    import datetime

    # We will look at stock prices over the past year, starting at January 1, 2016
    start = datetime.datetime(year_str, month_str, date_str)  # YY,MM,DD
    end = datetime.date.today()

    # Let's get Apple stock data; Apple's ticker symbol is AAPL
    # First argument is the series we want, second is the source ("yahoo" for Yahoo! Finance), third is the start date, fourth is the end date
    apple = web.DataReader(comp_str, "yahoo", start, end)
    type(apple)
    #print(apple.head())
    #print(apple.tail())
    apple.to_csv('f:/django/aapl_2_data.csv')

    dat = pd.read_csv('f:/django/aapl_2_data.csv')
    # X = dataset.iloc[:, 1:2].values
    # y = dataset.iloc[:, 2].values

    from datetime import datetime

    dataset = pd.read_csv('f:/django/aapl_data.csv')
    #print(X)

        #
        # dt = datetime(x)
        # milliseconds = int(round(dt.timestamp() * 1000))
        # x = milliseconds
        # print(milliseconds)
        #

    """
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    plt.scatter(X, y, color='red')
    plt.plot(X, lin_reg.predict(X), color='blue')
    plt.title('Truth or Bluff (Linear Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    """

    from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY
    import mpl_finance as mpf
    from matplotlib.dates import date2num

    def pandas_candlestick_ohlc(dat, stick="day", otherseries=None):
        """
        :param dat: pandas DataFrame object with datetime64 index, and float columns "Open", "High", "Low", and "Close", likely created via DataReader from "yahoo"
        :param stick: A string or number indicating the period of time covered by a single candlestick. Valid string inputs include "day", "week", "month", and "year", ("day" default), and any numeric input indicates the number of trading days included in a period
        :param otherseries: An iterable that will be coerced into a list, containing the columns of dat that hold other series to be plotted as lines

        This will show a Japanese candlestick plot for stock data stored in dat, also plotting other series if passed.
        """
        mondays = WeekdayLocator(MONDAY)  # major ticks on the mondays
        alldays = DayLocator()  # minor ticks on the days
        dayFormatter = DateFormatter('%d')  # e.g., 12

        # Create a new DataFrame which includes OHLC data for each period specified by stick input
        transdat = dat.loc[:, ["Open", "High", "Low", "Close"]]
        if (type(stick) == str):
            if stick == "day":
                plotdat = transdat
                stick = 1  # Used for plotting
            elif stick in ["week", "month", "year"]:
                if stick == "week":
                    transdat["week"] = pd.to_datetime(transdat.index).map(
                        lambda x: x.isocalendar()[1])  # Identify weeks
                elif stick == "month":
                    transdat["month"] = pd.to_datetime(transdat.index).map(lambda x: x.month)  # Identify months
                transdat["year"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[0])  # Identify years
                grouped = transdat.groupby(list(set(["year", stick])))  # Group by year and other appropriate variable
                plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [],
                                        "Close": []})  # Create empty data frame containing what will be plotted
                for name, group in grouped:
                    plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0, 0],
                                                           "High": max(group.High),
                                                           "Low": min(group.Low),
                                                           "Close": group.iloc[-1, 3]},
                                                          index=[group.index[0]]))
                if stick == "week":
                    stick = 5
                elif stick == "month":
                    stick = 30
                elif stick == "year":
                    stick = 365

        elif (type(stick) == int and stick >= 1):
            transdat["stick"] = [np.floor(i / stick) for i in range(len(transdat.index))]
            grouped = transdat.groupby("stick")
            plotdat = pd.DataFrame(
                {"Open": [], "High": [], "Low": [],
                 "Close": []})  # Create empty data frame containing what will be plotted
            for name, group in grouped:
                plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0, 0],
                                                       "High": max(group.High),
                                                       "Low": min(group.Low),
                                                       "Close": group.iloc[-1, 3]},
                                                      index=[group.index[0]]))

        else:
            raise ValueError(
                'Valid inputs to argument "stick" include the strings "day", "week", "month", "year", or a positive integer')

        # Set plot parameters, including the axis object ax used for plotting
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2)
        if plotdat.index[-1] - plotdat.index[0] < pd.Timedelta('730 days'):
            weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
            ax.xaxis.set_major_locator(mondays)
            ax.xaxis.set_minor_locator(alldays)
        else:
            weekFormatter = DateFormatter('%b %d, %Y')
        ax.xaxis.set_major_formatter(weekFormatter)

        ax.grid(True)

        # Create the candelstick chart
        mpf.candlestick_ohlc(ax, list(
            zip(list(date2num(plotdat.index.tolist())), plotdat["Open"].tolist(), plotdat["High"].tolist(),
                plotdat["Low"].tolist(), plotdat["Close"].tolist())),
                             colorup="black", colordown="red", width=stick * .4)

        # Plot other series (such as moving averages) as lines
        if otherseries != None:
            if type(otherseries) != list:
                otherseries = [otherseries]
            dat.loc[:, otherseries].plot(ax=ax, lw=1.3, grid=True)

        ax.xaxis_date()
        ax.autoscale_view()
        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

        plt.show()

    def simple_graph():
        # Control the default size of figures in this Jupyter notebook
        from pylab import rcParams
        rcParams['figure.figsize'] = (15, 9)  # Change the size of plots

        apple["Adj Close"].plot(grid=True)  # Plot the adjusted closing price of AAPL
        # plt.scatter()
        plt.ylabel("Stock Value")
        plt.xlabel(str(year_str) + "/" + str(month_str))
        plt.title("Stock of " + comp_str + " By Yahoo Finance")
        plt.show()


    if (graph_type == "simple"):
        simple_graph()
    elif (graph_type == "candle"):
        pandas_candlestick_ohlc(apple)

