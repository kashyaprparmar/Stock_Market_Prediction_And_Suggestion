from yahoo_fin import stock_info as si
# get live price of Apple
print(si.get_live_price("aapl"))
 
# or Amazon
print(si.get_live_price("amzn"))
 
# or any other ticker
#si.get_live_price(ticker)
