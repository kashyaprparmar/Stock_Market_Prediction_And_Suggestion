import urllib
import re
import time
import sys

def fetchstockquotes(symbol):
    base_url = 'http://finance.google.com/finance?q='
    content = urllib.urlopen(base_url + symbol).read()
    m = re.search('id="ref_(.*?)">(.*?)<', content)
    if m:
        quote = m.group(2)
    else:
        quote = 'no quote available for: ' + symbol
    return quote

symbol=sys.argv[0]
while True:
    print (str(time.ctime())+ " - " + symbol+ " - " + fetchstockquotes(symbol))
    time.sleep(5)
