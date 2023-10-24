import pandas as pd
import yfinance as yf
from datetime import *
import numpy as np
import yoptions as yo

"""
def expiries(ticker):
    df = yo.get_expiration_dates(stock_ticker=self.ticker)
    return df
"""
class Op:
    def __init__(self,K : float,ticker : str,N : int,ot :str,r=0.05,d=0.0,s=0.0):
        self.K = K
        self.ticker = ticker
        self.r = r
        self.d = d
        self.s = s
        self.N = N
        self.ot = ot

    def D(self):
        ticker = yf.Ticker(self.ticker)
        df = ticker.history(period="1y")['Close']
        df = pd.DataFrame(np.array(df.iloc[:]), columns=[self.ticker], index=np.array(df.index.strftime('%Y-%m-%d')))
        self.df = df
    def expiry(self,exp):
        self.T = (datetime.strptime(exp, '%Y-%m-%d').date() - date.today()).days / 365
    def dividend(self):
        ticker = yf.Ticker(self.ticker)
        c = float(sum(ticker.history(period="1y")['Dividends'])/self.df.iloc[-1])
        self.d = c
    def volatility(self):
        df1 = self.df.pct_change()
        df1 = df1.dropna()
        self.s = np.std(np.array(df1))*np.sqrt(256)


