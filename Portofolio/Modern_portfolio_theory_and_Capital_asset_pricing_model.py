import matplotlib.pyplot as plt
import numpy as np

from basic_functions_in_finance import *


#monte carlo simulation for the portfolio we use PCR to define dataframe df1
#use
def MCS_P(df,n_p=5000):
    n = len(df.columns)
    K = np.zeros((n_p, 2))
    d = (df.index[-1] - df.index[0]).days
    if (df.index[1] - df.index[0]).days > 3:
        d=12
    else:
        if d >= 252:
            d = 252
    r = R_P(df)
    S = C_M_P(df)
    weight = np.zeros((n_p, n))
    for i in range(n_p):
        weights = np.random.dirichlet(np.ones(n))
        K[i][0] = np.dot(weights, r)*d
        K[i][1] = np.sqrt(np.dot(np.dot(weights,S),np.transpose(weights))*d)
        weight[i] = weights
    K = pd.DataFrame(K, columns=['Return', 'Std'])
    weight = pd.DataFrame(weight, columns=df.columns)
    K = pd.concat([K, weight], axis=1)
    K = K.sort_values(by='Std', ascending=True, ignore_index=True)
    return K

#minimum variance of portfolio we use PCR to define dataframe df1
def MV_P(df):
    a = df[df['Std'] == min(df.Std)]
    return a
def MR_P(df):
    a = df[df['Return'] == max(df.Return)]
    return a
#theorical efficient frontier of portfolio
def T_F_E_P(DF, n_p=5000):
    DF = DF.dropna()
    n = len(DF.columns)
    d = (DF.index[-1] - DF.index[0]).days
    if (DF.index[1] - DF.index[0]).days > 3:
        d = 12
    else:
        if d >= 252:
            d = 252
    R = np.array(C_M_P(DF)) * d
    r = np.array(R_P(DF)) * d
    R1 = np.linalg.inv(R)
    I = np.reshape([1] * n, (n, 1))
    A = float(np.dot(np.dot(np.transpose(I), R1), r))
    D = float(np.dot(np.dot(np.transpose(r), R1), I))
    B = float(np.dot(np.dot(np.transpose(r), R1), r))
    C = float(np.dot(np.dot(np.transpose(I), R1), I))
    O = np.zeros((n_p, 2))
    a=0
    l = 1 / n_p
    weight = np.zeros((n_p, n))
    for i in np.arange(0, 1, l):
        beta = (2*i*A-2*B)/(D*A-C*B)
        lam = (2*i/B) - beta *(D/B)
        weights = np.reshape((1/2*(np.dot(R1,(lam*r+beta*I)))), (1, n))
        weight[a] = weights
        O[a][1] = np.sqrt(np.dot(np.dot(weights, R), np.transpose(weights)))
        O[a][0] = np.dot(weights, r)
        a = a + 1
    O = pd.DataFrame(O, columns=['Return', 'Std'])
    weight = pd.DataFrame(weight, columns=DF.columns)
    O = pd.concat([O, weight], axis=1)
    """
    for i in DF.columns:
        O=O[O[i] >= 0]
    O = O[O['Return'] > 0]
    O = O[O['Std'] > 0]
    """
    O = O.sort_values(by='Return', ascending=True, ignore_index=True)
    O = O.drop_duplicates()
    O = O.reset_index(drop=True)
    O = O.dropna()
    return O
#plot of MCS_P
def P_MCS_P(df1,n_p=5000):
    M = MCS_P(df1, n_p)
    plt.scatter(M.Std, M.Return,s=10,alpha=0.1)
    plt.xlim([0, max(M.Std) + 0.5])
    plt.title("Monte Carlo Simulation for the Portfolio")
    plt.xticks(rotation=45)
    plt.ylabel('Return')
    plt.xlabel("Volatility")
    plt.ylim([0, max(M.Return) + 0.5])
    return
#plot of MV_P
def P_MV_P(df1,n_p=5000):
    M = MCS_P(df1, n_p)
    a = MV_P(M)
    plt.scatter(M.Std, M.Return, s=10, alpha=0.1)
    plt.scatter(a.Std, a.Return, marker='*', s=100, label = 'Optimal portfolio by minimum variance')
    plt.legend(fontsize="7", loc="upper left")
    plt.xlim([0, max(M.Std) + 0.5])
    plt.title("Minimum Variance Portfolio")
    plt.xticks(rotation=45)
    plt.ylabel('Return')
    plt.xlabel("Volatility")
    plt.ylim([0, max(M.Return) + 0.5])
#theorical plot of T_F_E_P
def T_P_F_E_P(df1,n_p=5000):
    K= T_F_E_P(df1, n_p)
    M = MCS_P(df1, n_p)
    plt.plot(K.Std, K.Return,c='red')
    plt.scatter(M.Std, M.Return,s=10,alpha=0.1)
    plt.ylim([0, max(M.Return) + 0.1])
    plt.xlim([0,max(M.Std)+0.1])
    plt.ylabel("Return")
    plt.xlabel("Volatility")
    plt.title('Theorical Efficient Frontiere of Portfolio')


#reel sharpe ratio of portfolio
def S_R_P(df1,n_p=5000,rf=0.05):
    K = MCS_P(df1, n_p)
    K['SR'] = (K.Return - rf) / K.Std
    return K
#optimal portfolio with reel sharpe ratio
def O_S_R_P(df1,n_p=5000,rf=0.05):
    K = S_R_P(df1, n_p,rf)
    K = K[K['SR'] == max(K.SR)]
    return K
#plot of S_R_P
def P_S_R_P(df1,n_p=5000,rf=0.05):
    K = S_R_P(df1,n_p,rf)
    O = T_F_E_P(df1,n_p)
    a=K[K['SR']==max(K.SR)]
    b= K[K['Std']==min(K.Std)]
    plt.plot(O.Std, O.Return)
    plt.scatter(K.Std, K.Return, c=K.SR, s=7, cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(a.Std, a.Return, marker='*', s=100, label = 'Optimal portfolio by Sharpe Ratio')
    plt.scatter(b.Std, b.Return, marker='*', s=100, label = 'Optimal portfolio by minimum variance')
    plt.ylim([0, max(K.Return) + 0.1])
    plt.xlim([0, max(K.Std) + 0.1])
    plt.legend(fontsize="7", loc="upper left")
    plt.ylabel("Return")
    plt.xlabel("Volatility")
    plt.title('Sharpe Ratio And Optimum Portfolio')
#theorical plot of Capital Market Line CML of portfolio
def T_P_CML(df1,n_p=5000,rf=0.05):
    O = S_R_P(df1, n_p, rf)
    a = O[O['SR'] == max(O.SR)]
    plt.scatter(O.Std,O.Return,c=O.SR, s=7, cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.ylim([0, max(O.Return) + 0.1])
    plt.xlim([0, max(O.Std) + 0.1])
    plt.scatter(a.Std, a.Return, marker='*', s=100, label = 'Optimal portfolio by Sharpe Ratio')
    plt.legend(fontsize="7", loc="upper left")
    x=np.linspace(0,max(O.Std)+0.3,50)
    plt.plot(x,x*(a.iloc[0,0]-rf)/a.iloc[0,1]+rf,c='red')
    plt.ylabel("Return")
    plt.xlabel("Volatility")
    plt.title('Capital Market Line')
#reel efficient frontiere of portfolio
def R_E_F_P(df1,n_p=500):
    K = S_R_P(df1,n_p)
    MVP = K
    j=0
    for i in range(n_p):
        j = i + 1
        if j==n_p:
            break
        if MVP.iloc[i,0] > K.iloc[j,0] and MVP.iloc[i,1] < K.iloc[j,1]:
            for k in range(len(K.columns)):
                K.iloc[j, k] = MVP.iloc[i, k]
            i=j
    K = K[K['Return'] > 0]
    K = K.drop_duplicates(ignore_index=True)
    K = K.sort_values(by='Std')
    return K
#interpolation of points of the efficient frontiere
def I_R_F_E_P(df1,n_p=5000):
    K = R_E_F_P(df1,n_p)
    O = K[K['SR']==max(K.SR)]
    D = K[K['Std']<= O.iloc[0,1]].copy()
    N = len(D.index)
    X = np.zeros((4,N))
    Y = np.zeros((1,N))
    X[0] = [1]*N
    X[1] = D.iloc[:, 1]
    X[2] = np.log(D.iloc[:,1])
    X[3] = np.exp(D.iloc[:,1])
    Y[0] = D.iloc[:, 0]
    X = X.T
    Y = Y.T
    teta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Y)
    D['O_P'] = np.dot(X, teta)
    M = K[K['Std']> O.iloc[0,1]].copy()
    N = len(M.index)
    X = np.zeros((3, N))
    Y = np.zeros((1, N))
    X[0] = [1] * N
    X[1] = M.iloc[:, 1]
    X[2] = np.log(M.iloc[:, 1])
    Y[0] = M.iloc[:, 0]
    X = X.T
    Y = Y.T
    teta1 = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    M['O_P'] = np.dot(X,teta1)
    F = pd.concat([D, M])
    return F
#plot of R_F_E_P and SR
def P_R_E_F(df1,n_p=5000):
    K=I_R_F_E_P(df1,n_p)
    MVP = S_R_P(df1,n_p)
    a = K[K['SR'] == max(K.SR)]
    b= K[K['Std']==min(K.Std)]
    plt.plot(K.Std, K.Return)
    plt.scatter(MVP.Std, MVP.Return, c=MVP.SR, s=7, cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(a.Std, a.Return, marker='*', s=100, label = 'Optimal portfolio by Sharpe Ratio')
    plt.scatter(b.Std, b.Return, marker='*', s=100, label = 'Optimal portfolio by minimum variance')
    plt.legend(fontsize="7", loc="upper left")
    plt.ylim([0, max(MVP.Return) + 0.1])
    plt.xlim([0, max(MVP.Std) + 0.1])
    plt.ylabel("Return")
    plt.xlabel("Volatility")
    plt.title('Sharpe Ratio And Optimum Portfolio')
#plot of reel capital market line CML
def P_R_P_CML(df1,n_p=1000,rf=0.05):
    P_R_E_F(df1,n_p)
    O = S_R_P(df1,n_p,rf)
    a = O[O['SR'] == max(O.SR)]
    x = np.linspace(0, max(O.Std), 50)
    plt.plot(x, x * (a.iloc[0, 0] - rf) / a.iloc[0, 1] + rf, c='red')
#semivariance and sortiono Ratio of portfolio
def SO_R_P(df, n_p=5000,rf=0.05):
    n = len(df.columns)
    df1 = df.copy()
    X = np.zeros((n_p,4))
    weight = np.zeros((n_p, n))
    d = (df1.index[-1] - df1.index[0]).days
    if (df.index[1] - df.index[0]).days > 3:
        d = 12
    else:
        if d >= 252:
            d = 252
    S = C_M_P(df1)*d
    for i in range(n_p):
        weights = np.random.dirichlet(np.ones(n))
        X[i][1] = np.sqrt(np.dot(np.dot(weights, S), weights.T))
        df1['Portfolio'] = np.dot(df1, weights)
        weights = weights.T
        weight[i] = weights
        X[i][0] = R_P(df1).iloc[-1] * d
        R = STD_P(df1).iloc[-1] * np.sqrt(d)
        X[i][3] = (X[i][0] - rf) / R
        df1 = df1[df1['Portfolio'] < (X[i][0]/d)]
        R = STD_P(df1).iloc[-1] * np.sqrt(d)
        X[i][2] = (X[i][0]-rf)/R
        df1 = df.copy()
    X = pd.DataFrame(X, columns=['Return', 'Std','So_R','SR'])
    weight = pd.DataFrame(weight, columns=df1.columns)
    K = pd.concat([X, weight], axis=1)
    X = X.sort_values(by='Std', ascending=True, ignore_index=True)
    return K
#optimal portfolio with Sortino ratio
def O_SO_R_P(df1,n_p=5000,rf=0.05):
    K = SO_R_P(df1, n_p,rf)
    K = K[K['So_R'] == max(K.So_R)]
    return K
# plot of R_So_P
def P_SO_R_P(df1,n_p=5000,rf=0.05):
    O = SO_R_P(df1,n_p,rf)
    a = O[O['So_R'] == max(O.So_R)]
    b = O[O['Std']==min(O.Std)]
    plt.scatter(O.Std, O.Return, c=O.So_R, s=7, cmap='viridis')
    plt.colorbar(label='Sortino Ratio')
    plt.scatter(a.Std, a.Return, marker='*',color = 'blue' ,s=100, label = 'Optimal portfolio by Sortino Ratio')
    plt.scatter(b.Std, b.Return, marker='*',color = 'red' ,s=100, label = 'Optimal portfolio by minimum variance')
    plt.legend(fontsize="7", loc="upper left")
    plt.ylim([0, max(O.Return) + 0.1])
    plt.xlim([0, max(O.Std) + 0.1])
    plt.ylabel("Return")
    plt.xlabel("Volatility")
    plt.title('Sortino Ratio & Sharpe Ratio And Optimum Portfolio')
#Max drawdown calculate volatility and Calmer ratio is the return on the max draw down it s like how we expect for a return to be volitile of Portfolio
def MDR_CR_P(tickers, weights,interval,start = '2020-01-02',end = '2023-04-02',rf=0.05):
    DF = P(tickers,start,end,interval)
    n = len(DF.columns)
    weights = np.array(weights)
    M = np.dot(DF, weights.T)
    M = float((max(M) - min(M))/max(M))
    d = (DF.index[-1] - DF.index[0]).days
    if (DF.index[1] - DF.index[0]).days > 3:
        d = 12
    else:
        if d >= 252:
            d = 252
    R = float(np.dot(weights,np.array(R_P(P_C_R(DF))*d-rf)))
    C = R / M
    return M,C
#beta of portfolio
def beta_P(ticker,start,end,interval,n_p=5000):
    df1 = P_C_R(P(ticker,start,end,interval))
    n = len(df1.columns)
    df1['Nasdaq'] = P_C_R(P(['^IXIC'],start,end,interval))
    df1 = df1.dropna()
    d = (df1.index[-1] - df1.index[0]).days
    if (df1.index[1] - df1.index[0]).days > 3:
        d = 12
    else:
        if d >= 252:
            d = 252
    M = C_M_P(df1)
    S = float(STD_P(df1['Nasdaq']))**2
    K = np.array(M.iloc[:,-1]/S)
    MS = pd.DataFrame(K, columns=['Beta'], index=df1.columns)
    MS['Return'] = np.array(R_P(df1))*d
    a = MS.iloc[-1,-1]
    MS = MS.drop('Nasdaq')
    df1 = df1.iloc[:,:-1]
    Y = MCS_P(df1,n_p)
    Y['Beta'] = np.dot(Y.iloc[:,2:n+2],MS.Beta)
    Y = Y.dropna()
    return Y,a,MS
    
def beta_P1(ticker,start,end,interval,weights):
    df = P_C_R(P(ticker, start, end,interval))
    n = len(df.columns)
    weights = np.array(weights)
    if np.shape(weights)[0]==1:
        weights = np.transpose(weights)
    df1 = pd.DataFrame(np.dot(df,weights),columns=['Portfolio'],index=df.index)
    n = len(df1.columns)
    df1['Nasdaq'] = P_C_R(P(['^IXIC'], start, end,interval))
    df1 = df1.dropna()
    d = (df1.index[-1] - df1.index[0]).days
    if (df1.index[1] - df1.index[0]).days > 3:
        d = 12
    else:
        if d >= 252:
            d = 252
    M = C_M_P(df1)
    S = float(STD_P(df1['Nasdaq'])) ** 2
    K = np.array(M.iloc[:, -1] / S)
    MS = pd.DataFrame(K, columns=['Beta'], index=df1.columns)
    MS['Return'] = np.array(d*R_P(df1['Portfolio']))
    MS['Std'] = float(np.sqrt(d)*STD_P(df1['Portfolio']))
    a = MS.iloc[-1, -1]
    MS = MS.drop('Nasdaq')
    return MS
#plot security market line
def P_S_M_L(ticker,start,end,interval,n_p=5000,rf=0.05):
    Y,a,MS = beta_P(ticker,start,end,interval,n_p)
    B = np.linspace(-1,max(MS.Beta)+0.2,10)
    Y2 = rf + (a-rf) * B
    plt.plot(B, Y2,c='blue')
    colors = np.array(MS.Return)
    plt.scatter(MS.Beta,MS.Return , c=colors, cmap='viridis' ,s=20)
    plt.colorbar()
    for i in range(len(MS.index)):
        plt.text(MS.Beta[i], MS.Return[i], MS.index[i], va='bottom', ha='center')
    plt.axvline(x=0, c="black",alpha=0.5)
    plt.axhline(y=0, c="black",alpha=0.5)
    plt.ylim([-1, max(MS.Return) + 0.2])
    plt.xlim([-1, max(MS.Beta) + 0.2])
    plt.ylabel("Return")
    plt.xlabel("Beta")
    plt.title('Security Market Line')
#treynor ratio
def T_R_P(ticker,start,end,interval,n_p=5000,rf=0.05):
    Y,a,MS=beta_P(ticker,start,end,interval,n_p)
    Y['T_R'] = (Y.Return - rf) / Y.Beta
    return Y
#Optimal Portfolio with Treynor Ratio
def O_T_R_P(ticker,start,end,interval,n_p=5000,rf=0.05):
    Y = T_R_P(ticker,start,end,interval,n_p,rf)
    a = Y[Y['T_R'] == max(Y.T_R)]
    return a
#plot Treynor Ratio
def P_T_R_P(ticker,start,end,interval,n_p=5000,rf=0.05):
    Y = T_R_P(ticker,start,end,interval,n_p,rf)
    a = Y[Y['T_R'] == max(Y.T_R)]
    b= Y[Y['Std']==min(Y.Std)]
    plt.scatter(Y.Std, Y.Return, c=Y.T_R, s=7, cmap='viridis')
    plt.colorbar(label='Treynor Ratio')
    plt.scatter(a.Std, a.Return, marker='*', s=100)
    plt.scatter(b.Std, b.Return, marker='*', s=100)
    plt.ylim([0, a.iloc[0,0]+0.1])
    plt.xlim([0, max(Y.Std) + 0.1])
    plt.ylabel("Return")
    plt.xlabel("Volatility")
    plt.title('Treynor Ratio And Optimum Portfolio')
#  'WMT','AMZN','XOM','NVDA','META','HD','A','AA'
#Alpha de Jensen
def A_J_P(ticker,start,end,interval,weights,n_p=5000,rf=0.05):
    Y,a,MS = beta_P(ticker, start, end,interval, n_p)
    Y = beta_P1(ticker, start, end, interval, weights)
    Y['A_J'] = Y.Return - (rf + Y.Beta * (a - rf))
    MS['A_J'] = MS.Return - (rf + MS.Beta * (a - rf))
    return Y,MS





