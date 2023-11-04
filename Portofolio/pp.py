import pandas as pd
import numpy as np 
from pc import *

def MCS(df) :
    
    n = len(df.columns)
    sm = 5000
    weights=[]
    K = np.zeros((sm,2))


    for i in range(sm) : 
        weight = np.random.random(n)
        weight = weight/np.sum(weight)
        weights.append(weight)
        K[i][0] = float(np.dot(weight,df.sum()))
        K[i][1] = float(np.sqrt(np.dot(np.dot(np.transpose(weight),df.cov()),weight)*df.count().values[0]))
   
    result= pd.DataFrame(K,columns=["return","std"])
    sharpe_ratio = pd.DataFrame((result["return"] - 0.05) / result["std"],columns=["Sharpe Ratio"])
    weights = pd.DataFrame(weights,columns=df.columns)
    result= pd.concat([result,sharpe_ratio,weights],axis=1)
    result.sort_values(by="std",ascending=True,ignore_index=True, inplace=True)

    

    return result

def minVar(df) :
    return df.iloc[0]


def plotGraph(df) :
    slope = df["Sharpe Ratio"].max()

    x = np.linspace(0,0.5,50) 
    y = 0.05 + slope*x

    df.plot.scatter("std", "return", marker='o', grid=True, c=df["Sharpe Ratio"], cmap='inferno', title='Portfolio Perfomance', figsize=(10,5))
    plt.plot(minVar(df)["std"],minVar(df)["return"], '*', markersize=7, c='k', label="Minimum Variance portfolio")
    plt.plot(df["std"].iloc[df["Sharpe Ratio"].idxmax()], df["return"].iloc[df["Sharpe Ratio"].idxmax()], '*', markersize=7, c='g', label="Market portfolio")

    plt.xlabel('Portfolio risk')
    plt.ylabel('Expected return')
    plt.plot(x,y, label='CML')
    plt.ylim(ymin=0, ymax=0.45)
    plt.xlim(xmin=0, xmax=0.5)
    plt.legend()
    plt.show()
