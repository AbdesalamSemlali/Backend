import yfinance as yf
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


def P(tickers,start,end) : 
    df = yf.download(tickers=tickers,start=start,end=end)["Adj Close"]
    return df

def R(df) : 
    df = df.pct_change().dropna()
    return df

def RSTD(df) :
    d = (df.index[-1]- df.index[0]).days
    if d >= 256 :
        d = 256 
    return df.mean()*d,df.std()*np.sqrt(d)


def PCM(df,f) : 
    plt.figure(figsize=(10,10))
    sns.heatmap(df.corr(),vmin=-1,vmax=1,linewidths=1,annot=True,xticklabels=df.columns,yticklabels=df.columns)
    plt.title("Portfolio Components Correlation Matrix")
    plt.savefig(f,format="png")


#print(CM(R(P(["AAPL","GOOGL","TSLA"],"2022-11-01","2023-11-01"))))


