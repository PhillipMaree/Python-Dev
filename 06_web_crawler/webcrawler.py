'''
Created on 09 Aug 2019

@author: maree
'''
import requests
from bs4 import BeautifulSoup
import pandas as pd
import operator

class WebCrawlerC():
    '''
    classdocs
    '''

    def __init__(self):
        '''
        WebCrawlerC constructor
        '''
        
        return
    
    def get_html(self, ticker):
        '''
        Return the bautified html
        '''
        
        url = "https://finance.yahoo.com/quote/" + ticker + "?p=" + ticker + "&.tsrc=fin-srch"
        page = requests.get( url )
        html = BeautifulSoup(page.content, 'html.parser')
        
        return html
    
    def get_stock_quote(self, ticker): 
        '''
        Retrieves the current stock price
        '''
         
        html = self.get_html(ticker)
        temp = html.select('#quote-header-info > div.My\(6px\).Pos\(r\).smartphone_Mt\(6px\) > div')
        quote = list(temp[0].stripped_strings)[0]
        
        return quote
    
    def get_quote_summary(self, ticker):        
        '''
        Retrieves the table summarizing stock movements and performance
        '''
        html = self.get_html(ticker)
        labels = list( map(operator.attrgetter("text"), html.find_all(class_='C(black) W(51%)')) )
        values = list( map(operator.attrgetter("text"), html.find_all(class_='Ta(end) Fw(600) Lh(14px)'))  )
        summary = pd.Series(values, labels)
        
        return summary
        