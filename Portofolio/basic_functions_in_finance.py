import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pylab as plt
import seaborn as sns

#Portfolio components
#use
def P(tickers,start='2022-03-13',end='2023-03-13',interval = "1d",price_type = 'Adj Close'):
    n = len(tickers)
    if (n == 1) or (isinstance(tickers, str)):
        df = yf.download(tickers=tickers, start=start, end=end,interval = interval)[price_type]
        columns = []
        columns.append(tickers)
    else :
        df = yf.download(tickers=tickers, start=start, end=end,interval = interval)[price_type]
        columns = tickers
    df = pd.DataFrame(df)
    N = int(len(df.index))
    n = int(len(df.columns))
    rows = []
    for i in range(N):
        rows.append(df.iloc[i, :])
    rows = np.array(rows)
    df = pd.DataFrame(rows,columns=columns,index = np.array(df.index))
    df = df.dropna()
    return df
#Portfolio components return we use P to define dataframe df
#use

def P_C_R(df):
    N = int(len(df.index))
    df1 = df.copy()
    if isinstance(df, pd.Series):
        n = 1
        df1 = df.pct_change()
    else:
        n = len(df.columns)
        for i in range(n):
            df1.iloc[:, i] = df.iloc[:, i].pct_change()
    df1 = df1.dropna()
    return df1
#Components portfolio plot we use P to define dataframe df
def P_P(df):
    if isinstance(df, pd.Series):
        n = 1
    else:
        n = len(df.columns)
    if n == 1:
        plt.plot(df.index, df, label = df.columns[0])
    elif n > 1:
        for i in range(n):
            plt.plot(df.index,df.iloc[:,i],label = df.columns[i],alpha=0.5)
    plt.xticks(rotation=45)
    plt.legend()
    plt.ylabel('Price')
    plt.title("Tickers plot by date")
#Return portfolio we use PCR to define dataframe df1
#use
def R_P(df):
    if isinstance(df, pd.DataFrame):
        n = len(df.columns)
        r = [None] * n
        for i in range(n):
            r[i] = sum(df.iloc[:, i]) / len(df.index)
        r = pd.DataFrame(r, columns=['Mean'], index=df.columns)
    else:
        p = list(np.shape(df))
        if len(p) == 1:
            p.append(1)
        r = [None] * p[1]
        r = sum(df) / p[0]
    return r
#STD portfolio we use PCR to define dataframe df1
#use
def STD_P(df):
    mean = np.array(R_P(df))
    if isinstance(df, pd.DataFrame):
        n = len(df.columns)
        N = len(df.index)
        sigma = [None] * n
        for i in range(n):
            sigma[i] = np.sqrt(np.dot((df.iloc[:, i] - mean[i]), (df.iloc[:, i] - mean[i])) / (N - 1))
        sigma = pd.DataFrame(sigma, columns=['Standard deviation'], index=df.columns)
    else:
        p = list(np.shape(df))
        if len(p) == 1:
            p.append(1)
        sigma = [None] * p[1]
        for i in range(p[1]):
            sigma[i] = np.sqrt(np.dot((df - mean), (df - mean)) / (p[0] - 1))
            sigma = np.array(sigma)
    return sigma
#covariance matrix of portfolio we use PCR to define dataframe df1
#use

def C_M_P(df):
    mean = np.array(R_P(df))
    if isinstance(df, pd.DataFrame):
        n = len(df.columns)
        N = len(df.index)
        MS = np.zeros((n, n))
        for i in range(n):
            MS[i][i] = np.array(STD_P(df))[i] ** 2
            for j in range(i + 1, n):
                MS[i][j] = np.dot(np.transpose(df.iloc[:, i] - mean[i]), (df.iloc[:, j] - mean[j])) / (N - 1)
                MS[j][i] = np.dot(np.transpose(df.iloc[:, i] - mean[i]), (df.iloc[:, j] - mean[j])) / (N - 1)
        MS = pd.DataFrame(MS, columns=df.columns, index=df.columns)
    else:
        p = list(np.shape(df))
        if len(p) == 1:
            p.append(1)
        MS = np.zeros((p[1],p[1]))
        for i in range(p[1]):
            MS[i][i] = np.array(STD_P(df))[i] ** 2
            for j in range(i + 1, p[1]):
                MS[i][j] = np.dot(np.transpose(df[:, i] - mean[i]), (df[:, j] - mean[j])) / (p[0] - 1)
                MS[j][i] = np.dot(np.transpose(df[:, i] - mean[i]), (df[:, j] - mean[j])) / (p[0] - 1)
    return MS
#Portfolio correlation we use PCR to define dataframe df1
#use

def Correlation(df):
    if isinstance(df, pd.DataFrame):
        n = len(df.columns)
        C = np.zeros((n, n))
        S = C_M_P(df)
        for i in range(n):
            for j in range(n):
                C[i][j] = S.iloc[i, j] / np.sqrt(S.iloc[i, i] * S.iloc[j, j])
        C = pd.DataFrame(C, columns=df.columns, index=df.columns)
    else:
        p = list(np.shape(df))
        if len(p) == 1:
            p.append(1)
        C = np.zeros((p[1], p[1]))
        S = C_M_P(df)
        for i in range(p[1]):
            for j in range(p[1]):
                C[i][j] = S[i, j] / np.sqrt(S[i, i] * S[j, j])
    return C
#Portfolio correlation plot we use PCR to define dataframe df1
#use
def P_Correlation(df1):
    R=Correlation(df1)
    plt.figure(figsize=(10, 10))
    heat_map = sns.heatmap(R, vmin=-1, vmax=1, linewidth=1, annot=True, xticklabels=df1.columns, yticklabels=df1.columns)
    plt.title("HeatMap Correlation")











