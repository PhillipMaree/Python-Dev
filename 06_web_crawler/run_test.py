from webcrawler import WebCrawlerC

obj = WebCrawlerC()
'''
print("Vestas stock price: "+ obj.get_stock_quote('VWS.CO'))
print("Apple stock price: "+ obj.get_stock_quote('AAPL'))
print("Tesla stock price: "+ obj.get_stock_quote('TSLA'))
'''
print( obj.get_quote_summary('AAPL') )