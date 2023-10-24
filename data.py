import yfinance as yf
from datetime import *
import pandas as pd
import numpy as np
import yoptions as yo

def P(ticker,start='2022-03-13',interval = "1d",price_type = 'Adj Close'):
    end = str(date.today())
    df = yf.download(tickers=ticker, start=start, end=end,interval = interval)[price_type]
    df = pd.DataFrame( np.array(df.iloc[:]),columns=[ticker],index=np.array(df.index))
    return df
def P_R(df):
    df1 = df.pct_change()
    df1 = df1.dropna()
    return df1
