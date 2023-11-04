import numpy as np
from scipy.stats import norm
from Classes import *
import matplotlib.pyplot as plt



class P_eOp(Op):
    def __init__(self,K : float,ticker : str,N : int,ot :str,exp : str,r : float,d : float,s : float):
        super().__init__(K,ticker,N,ot,exp,r,d,s)


    def BlackScholes(self,St: float, K: float, T: float, s: float, R: float, type: str, d: float):
        d1 = (1 / (s * np.sqrt(T))) * (np.log(St / K) + ((R - d) + 0.5 * (s) ** 2) * T)
        d2 = d1 - s * np.sqrt(T)
        if type == "Call":
            return np.exp(d * T) * St * norm.cdf(d1) - K * np.exp(-R * T) * norm.cdf(d2)
        elif type == "Put":
            return -np.exp(d * T) * St * norm.cdf(-d1) + K * np.exp(-R * T) * norm.cdf(-d2)

    def delta(self,St: float, K: float, T: float, Sigma: float, R: float, type: str, d : float):
        d1 = (1 / (Sigma * np.sqrt(T))) * (np.log(St / K) + ((R-d) + 0.5 * (Sigma) ** 2) * T)
        return np.exp(-d * T)*norm.cdf(d1) if type == "Call" else -np.exp(-d * T)*norm.cdf(-d1)

    def gamma(self,St: float, K: float, T: float, Sigma: float, R: float, type: str,d : float):
        d1 = (1 / (Sigma * np.sqrt(T))) * (np.log(St / K) + ((R-d) + 0.5 * (Sigma) ** 2) * T)
        return np.exp(-d * T)*norm.pdf(d1) / (St * Sigma * np.sqrt(T))

    def theta(self,St: float, K: float, T: float, Sigma: float, R: float, type: str,d : float):
        d1 = (1 / (Sigma * np.sqrt(T))) * (np.log(St / K) + ((R-d) + 0.5 * (Sigma) ** 2) * T)
        d2 = d1 - Sigma * np.sqrt(T)
        call = (-np.exp(-d * T)*St * norm.pdf(d1, 0, 1) * Sigma) / (2 * np.sqrt(T)) - R * K * np.exp(-R * T) * norm.cdf(d2, 0, 1) + d * St * np.exp(-d*T) * norm.cdf(d1, 0, 1)
        put = (-np.exp(-d * T)*St * norm.pdf(d1, 0, 1) * Sigma) / (2 * np.sqrt(T)) + R * K * np.exp(-R * T) * norm.cdf(-d2) - d * St * np.exp(-d*T) * norm.cdf(-d1, 0, 1)
        return call if type == "Call" else put

    def vega(self,St: float, K: float, T: float, Sigma: float, R: float, type: str,d : float):
        d1 = (1 / (Sigma * np.sqrt(T))) * (np.log(St / K) + ((R-d) + 0.5 * (Sigma) ** 2) * T)
        d2 = d1 - Sigma * np.sqrt(T)
        return St * np.exp(-d * T) * norm.pdf(d1) * np.sqrt(T)


    def rho(self,St: float, K: float, T: float, Sigma: float, R: float, type: str,d : float):
        d1 = (1 / (Sigma * np.sqrt(T))) * (np.log(St / K) + ((R-d) + 0.5 * (Sigma) ** 2) * T)
        d2 = d1 - Sigma * np.sqrt(T)
        call = K * T * np.exp(-R * T) * norm.cdf(d2, 0, 1)
        put = -K * T * np.exp(-R * T) * norm.cdf(-d2, 0, 1)
        return call  if type == "Call" else put

    def P2_OP(self,x_min: float, x_max: float, x: str):
        x_min = int(x_min)
        x_max = int(x_max)

        X = ['Stock Price', 'Strike', 'Maturity', 'Volatility', 'Interest rate', 'Dividend']
        X = np.array(X)
        matching_indices = np.where(X == x)[0]

        if x_max<2:
            X = np.arange(x_min, x_max,0.1)
            Y = np.zeros((6,len(X)))
        else:
            X = np.arange(x_min, x_max)
            Y = np.zeros((6, len(X)))


        match matching_indices:
            case 0:
                p=0
                for i in X :
                    Y[0,p] = self.BlackScholes(i, self.K, self.T, self.s, self.r, self.ot, self.d)
                    Y[1, p] = self.delta(i, self.K, self.T, self.s, self.r, self.ot, self.d)
                    Y[2, p] = self.gamma(i, self.K, self.T, self.s, self.r, self.ot, self.d)
                    Y[3, p] = self.rho(i, self.K, self.T, self.s, self.r, self.ot, self.d)
                    Y[4, p] = self.theta(i, self.K, self.T, self.s, self.r, self.ot, self.d)
                    Y[5, p] = self.vega(i, self.K, self.T, self.s, self.r, self.ot, self.d)
                    p=p+1
                return X,Y

            case 1:
              p=0
              for i in X:
                  Y[0, p] = self.BlackScholes(self.df.iloc[-1,0],i, self.T, self.s, self.r, self.ot, self.d)
                  Y[1, p] = self.delta(self.df.iloc[-1,0], i, self.T, self.s, self.r, self.ot, self.d)
                  Y[2, p] = self.gamma(self.df.iloc[-1,0], i, self.T, self.s, self.r, self.ot, self.d)
                  Y[3, p] = self.rho(self.df.iloc[-1,0], i, self.T, self.s, self.r, self.ot, self.d)
                  Y[4, p] = self.theta(self.df.iloc[-1,0], i, self.T, self.s, self.r, self.ot, self.d)
                  Y[5, p] = self.vega(self.df.iloc[-1,0], i, self.T, self.s, self.r, self.ot,self.d)
                  p=p+1
              return X, Y

            case 2:
                p=0
                for i in X:
                    Y[0, p] = self.BlackScholes(self.df.iloc[-1, 0], self.K,i,self.s, self.r, self.ot, self.d)
                    Y[1, p] = self.delta(self.df.iloc[-1, 0], self.K,i,self.s, self.r, self.ot, self.d)
                    Y[2, p] = self.gamma(self.df.iloc[-1, 0], self.K,i, self.s, self.r, self.ot, self.d)
                    Y[3, p] = self.rho(self.df.iloc[-1, 0], self.K,i, self.s, self.r, self.ot, self.d)
                    Y[4, p] = self.theta(self.df.iloc[-1, 0], self.K,i, self.s, self.r, self.ot, self.d)
                    Y[5, p] = self.vega(self.df.iloc[-1, 0], self.K,i, self.s, self.r, self.ot,self.d)
                    p=p+1
                return X, Y

            case 3:
                p=0
                for i in X:
                    Y[0, p] = self.BlackScholes(self.df.iloc[-1, 0], self.K,self.T ,i, self.r, self.ot, self.d)
                    Y[1, p] = self.delta(self.df.iloc[-1, 0], self.K,self.T, i, self.r, self.ot, self.d)
                    Y[2, p] = self.gamma(self.df.iloc[-1, 0], self.K,self.T, i, self.r, self.ot, self.d)
                    Y[3, p] = self.rho(self.df.iloc[-1, 0], self.K, self.T,i, self.r, self.ot, self.d)
                    Y[4, p] = self.theta(self.df.iloc[-1, 0], self.K, self.T,i, self.r, self.ot, self.d)
                    Y[5, p] = self.vega(self.df.iloc[-1, 0], self.K, self.T,i, self.r, self.ot,self.d)
                    p=p+1
                return X, Y

            case 4:
                p=0
                for i in X:
                    Y[0, p] = self.BlackScholes(self.df.iloc[-1, 0], self.K, self.T, self.s, i, self.ot, self.d)
                    Y[1, p] = self.delta(self.df.iloc[-1, 0], self.K, self.T, self.s, i, self.ot, self.d)
                    Y[2, p] = self.gamma(self.df.iloc[-1, 0], self.K, self.T, self.s, i, self.ot, self.d)
                    Y[3, p] = self.rho(self.df.iloc[-1, 0], self.K, self.T, self.s, i, self.ot, self.d)
                    Y[4, p] = self.theta(self.df.iloc[-1, 0], self.K, self.T, self.s, i, self.ot, self.d)
                    Y[5, p] = self.vega(self.df.iloc[-1, 0], self.K, self.T, self.s, i, self.ot, self.d)
                    p=p+1
                return X, Y

            case 5:
                p=0
                for i in X:
                    Y[0, p] = self.BlackScholes(self.df.iloc[-1, 0], self.K, self.T, self.s, self.r, self.ot, i)
                    Y[1, p] = self.delta(self.df.iloc[-1, 0], self.K, self.T, self.s, self.r, self.ot, i)
                    Y[2, p] = self.gamma(self.df.iloc[-1, 0], self.K, self.T, self.s, self.r, self.ot, i)
                    Y[3, p] = self.rho(self.df.iloc[-1, 0], self.K, self.T, self.s, self.r, self.ot, i)
                    Y[4, p] = self.theta(self.df.iloc[-1, 0], self.K, self.T, self.s, self.r, self.ot, i)
                    Y[5, p] = self.vega(self.df.iloc[-1, 0], self.K, self.T, self.s, self.r, self.ot, i)
                    p=p+1
                return X, Y
            case _:
                print("error")
    def P3_OP(self,x_min: float, x_max: float,y_min: float, y_max: float, x: str, y: str, t: str):
        X = ['Stock Price', 'Strike', 'Maturity', 'Volatility', 'Interest rate', 'Dividend']
        X = np.array(X)
        m1 = int(np.where(X == x)[0])
        m2 = int(np.where(X == y)[0])
        matching_indices = str(m1)+str(m2)
        if (x_max-x_min)<(y_max-y_min):
            if x_max <= 10:
                X = np.arange(x_min, x_max,0.1)
                Y = np.arange(y_min, y_max,(y_max-y_min)/len(X))
            else:
                X = np.arange(x_min, x_max)
                Y = np.arange(y_min, y_max, (y_max - y_min) / len(X))
        else:
            if y_max <= 10:
                Y = np.arange(y_min, y_max,0.1)
                X = np.arange(x_min, x_max, (x_max - x_min) / len(Y))
            else:
                X = np.arange(x_min, x_max)
                Y = np.arange(y_min, y_max, (y_max - y_min) / len(X))
        Z = np.zeros((len(X),len(Y)))
        Q = np.arange(0,len(X)+1)


        match matching_indices:
            case '01' | '10':
                if matching_indices == '10':
                    u = x_min
                    x_min = y_min
                    y_min = u
                    u = x_max
                    x_max = y_max
                    y_max = u
                    T = X
                    X = Y
                    Y = T
                p = 0
                for i in X:
                    m = 0
                    for j in Y:
                        match t:
                            case 'BS':
                                Z[Q[p],Q[m]] = self.BlackScholes(i, j, self.T, self.s, self.r, self.ot, self.d)
                            case 'delta':
                                Z[Q[p],Q[m]] = self.delta(i, j, self.T, self.s, self.r, self.ot, self.d)
                            case 'gamma':
                                Z[Q[p],Q[m]] = self.gamma(i, j, self.T, self.s, self.r, self.ot, self.d)
                            case 'rho':
                                Z[Q[p],Q[m]] = self.rho(i, j, self.T, self.s, self.r, self.ot, self.d)
                            case 'theta':
                                Z[Q[p],Q[m]] = self.theta(i, j, self.T, self.s, self.r, self.ot, self.d)
                            case 'vega':
                                Z[Q[p],Q[m]] = self.vega(i, j, self.T, self.s, self.r, self.ot, self.d)

                        m = m+1
                    p = p+1
                return np.meshgrid(X,Y) , Z.transpose()

            case '02' | '20':
                if matching_indices == '20':
                    u = x_min
                    x_min = y_min
                    y_min = u
                    u = x_max
                    x_max = y_max
                    y_max = u
                    T = X
                    X = Y
                    Y = T
                p = 0
                for i in X:
                    m = 0
                    for j in Y:
                        match t:
                            case 'BS':
                                Z[Q[p],Q[m]] = self.BlackScholes(i, self.K, j, self.s, self.r, self.ot,self.d)
                            case 'delta':
                                Z[Q[p],Q[m]] = self.delta(i, self.K, j, self.s, self.r, self.ot,self.d)
                            case 'gamma':
                                Z[Q[p],Q[m]] = self.gamma(i, self.K, j, self.s, self.r, self.ot,self.d)
                            case 'rho':
                                Z[Q[p],Q[m]] = self.rho(i, self.K, j, self.s, self.r, self.ot,self.d)
                            case 'theta':
                                Z[Q[p], Q[m]] = self.theta(i, self.K, j, self.s, self.r, self.ot,self.d)
                            case 'vega':
                                Z[Q[p],Q[m]] = self.vega(i, self.K, j, self.s, self.r, self.ot,self.d)

                        m = m+1
                    p = p+1
                return np.meshgrid(X,Y) , Z.transpose()

            case '03' | '30':
                if matching_indices == '30':
                    u = x_min
                    x_min = y_min
                    y_min = u
                    u = x_max
                    x_max = y_max
                    y_max = u
                    T = X
                    X = Y
                    Y = T
                p = 0
                for i in X:
                    m = 0
                    for j in Y:
                        match t:
                            case 'BS':
                                Z[Q[p],Q[m]] = self.BlackScholes(i, self.K, self.T,j, self.r, self.ot,self.d)
                            case 'delta':
                                Z[Q[p],Q[m]] = self.delta(i, self.K, self.T,j, self.r, self.ot, self.d)
                            case 'gamma':
                                Z[Q[p],Q[m]] = self.gamma(i, self.K, self.T,j, self.r, self.ot, self.d)
                            case 'rho':
                                Z[Q[p],Q[m]] = self.rho(i, self.K, self.T,j, self.r, self.ot, self.d)
                            case 'theta':
                                Z[Q[p],Q[m]] = self.theta(i, self.K, self.T,j, self.r, self.ot, self.d)
                            case 'vega':
                                Z[Q[p],Q[m]] = self.vega(i, self.K, self.T,j, self.r, self.ot, self.d)

                        m = m+1
                    p = p+1
                return np.meshgrid(X,Y) , Z.transpose()

            case '04' | '40':
                if matching_indices == '40':
                    u = x_min
                    x_min = y_min
                    y_min = u
                    u = x_max
                    x_max = y_max
                    y_max = u
                    T = X
                    X = Y
                    Y = T
                p = 0
                for i in X:
                    m = 0
                    for j in Y:
                        match t:
                            case 'BS':
                                Z[Q[p],Q[m]] = self.BlackScholes(i, self.K, self.T, self.s, j, self.ot,self.d)
                            case 'delta':
                                Z[Q[p],Q[m]] = self.delta(i, self.K, self.T, self.s, j, self.ot,self.d)
                            case 'gamma':
                                Z[Q[p],Q[m]] = self.gamma(i, self.K, self.T, self.s, j, self.ot,self.d)
                            case 'rho':
                                Z[Q[p],Q[m]] = self.rho(i, self.K, self.T, self.s, j, self.ot,self.d)
                            case 'theta':
                                Z[Q[p],Q[m]] = self.theta(i, self.K, self.T, self.s, j, self.ot,self.d)
                            case 'vega':
                                Z[Q[p],Q[m]] = self.vega(i, self.K, self.T, self.s, j, self.ot,self.d)

                        m = m+1
                    p = p+1
                return np.meshgrid(X,Y) , Z.transpose()

            case '05' | '50':
                if matching_indices == '50':
                    u = x_min
                    x_min = y_min
                    y_min = u
                    u = x_max
                    x_max = y_max
                    y_max = u
                    T = X
                    X = Y
                    Y = T
                p = 0
                for i in X:
                    m = 0
                    for j in Y:
                        match t:
                            case 'BS':
                                Z[Q[p],Q[m]] = self.BlackScholes(i, self.K, self.T, self.s, self.r, self.ot,j)
                            case 'delta':
                                Z[Q[p],Q[m]] = self.delta(i, self.K, self.T, self.s, self.r, self.ot,j)
                            case 'gamma':
                                Z[Q[p],Q[m]] = self.gamma(i, self.K, self.T, self.s, self.r, self.ot,j)
                            case 'rho':
                                Z[Q[p],Q[m]] = self.rho(i, self.K, self.T, self.s, self.r, self.ot,j)
                            case 'theta':
                                Z[Q[p],Q[m]] = self.theta(i, self.K, self.T, self.s, self.r, self.ot,j)
                            case 'vega':
                                Z[Q[p],Q[m]] = self.vega(i, self.K, self.T, self.s, self.r, self.ot,j)

                        m = m+1
                    p = p+1
                return np.meshgrid(X,Y) , Z.transpose()

            case '21' | '12':
                if matching_indices == '21':
                    u = x_min
                    x_min = y_min
                    y_min = u
                    u = x_max
                    x_max = y_max
                    y_max = u
                    T = X
                    X = Y
                    Y = T
                p = 0
                for i in X:
                    m = 0
                    for j in Y:
                        match t:
                            case 'BS':
                                Z[Q[p],Q[m]] = self.BlackScholes(self.df.iloc[-1,0], i, j, self.s, self.r, self.ot,self.d)
                            case 'delta':
                                Z[Q[p],Q[m]] = self.delta(self.df.iloc[-1,0], i, j, self.s, self.r, self.ot,self.d)
                            case 'gamma':
                                Z[Q[p],Q[m]] = self.gamma(self.df.iloc[-1,0], i, j, self.s, self.r, self.ot,self.d)
                            case 'rho':
                                Z[Q[p],Q[m]] = self.rho(self.df.iloc[-1,0], i, j, self.s, self.r, self.ot,self.d)
                            case 'theta':
                                Z[Q[p],Q[m]] = self.theta(self.df.iloc[-1,0], i, j, self.s, self.r, self.ot,self.d)
                            case 'vega':
                                Z[Q[p],Q[m]] = self.vega(self.df.iloc[-1,0], i, j, self.s, self.r, self.ot,self.d)
                        m = m+1
                    p = p+1
                return np.meshgrid(X,Y) , Z.transpose()
            case '31' | '13':
                if matching_indices == '31':
                    u = x_min
                    x_min = y_min
                    y_min = u
                    u = x_max
                    x_max = y_max
                    y_max = u
                    T = X
                    X = Y
                    Y = T
                p = 0
                for i in X:
                    m = 0
                    for j in Y:
                        match t:
                            case 'BS':
                                Z[Q[p],Q[m]] = self.BlackScholes(self.df.iloc[-1,0], i, self.T, j, self.r, self.ot,self.d)
                            case 'delta':
                                Z[Q[p],Q[m]] = self.delta(self.df.iloc[-1,0], i, self.T, j, self.r, self.ot,self.d)
                            case 'gamma':
                                Z[Q[p],Q[m]] = self.gamma(self.df.iloc[-1,0], i, self.T, j, self.r, self.ot,self.d)
                            case 'rho':
                                Z[Q[p],Q[m]] = self.rho(self.df.iloc[-1,0], i, self.T, j, self.r, self.ot,self.d)
                            case 'theta':
                                Z[Q[p],Q[m]] = self.theta(self.df.iloc[-1,0], i, self.T, j, self.r, self.ot,self.d)
                            case 'vega':
                                Z[Q[p],Q[m]] = self.vega(self.df.iloc[-1,0], i, self.T, j, self.r, self.ot,self.d)
                        m = m+1
                    p = p+1
                return np.meshgrid(X,Y) , Z.transpose()
            case '41' | '14':
                if matching_indices == '41':
                    u = x_min
                    x_min = y_min
                    y_min = u
                    u = x_max
                    x_max = y_max
                    y_max = u
                    T = X
                    X = Y
                    Y = T
                p = 0
                for i in X:
                    m = 0
                    for j in Y:
                        match t:
                            case 'BS':
                                Z[Q[p],Q[m]] = self.BlackScholes(self.df.iloc[-1,0], i, self.T, self.s, j, self.ot,self.d)
                            case 'delta':
                                Z[Q[p],Q[m]] = self.delta(self.df.iloc[-1,0], i, self.T, self.s, j, self.ot,self.d)
                            case 'gamma':
                                Z[Q[p],Q[m]] = self.gamma(self.df.iloc[-1,0], i, self.T, self.s, j, self.ot,self.d)
                            case 'rho':
                                Z[Q[p],Q[m]] = self.rho(self.df.iloc[-1,0], i, self.T, self.s, j, self.ot,self.d)
                            case 'theta':
                                Z[Q[p],Q[m]] = self.theta(self.df.iloc[-1,0], i, self.T, self.s, j, self.ot,self.d)
                            case 'vega':
                                Z[Q[p],Q[m]] = self.vega(self.df.iloc[-1,0], i, self.T, self.s, j, self.ot,self.d)
                        m = m+1
                    p = p+1

                return np.meshgrid(X,Y) , Z.transpose()
            case '51' | '15':
                if matching_indices == '51':
                    u = x_min
                    x_min = y_min
                    y_min = u
                    u = x_max
                    x_max = y_max
                    y_max = u
                    T = X
                    X = Y
                    Y = T
                p = 0
                for i in X:
                    m = 0
                    for j in Y:
                        match t:
                            case 'BS':
                                Z[Q[p],Q[m]] = self.BlackScholes(self.df.iloc[-1,0], i, self.T, self.s, self.r, self.ot,j)
                            case 'delta':
                                Z[Q[p],Q[m]] = self.delta(self.df.iloc[-1,0], i, self.T, self.s, self.r, self.ot,j)
                            case 'gamma':
                                Z[Q[p],Q[m]] = self.gamma(self.df.iloc[-1,0], i, self.T, self.s, self.r, self.ot,j)
                            case 'rho':
                                Z[Q[p],Q[m]] = self.rho(self.df.iloc[-1,0], i, self.T, self.s, self.r, self.ot,j)
                            case 'theta':
                                Z[Q[p],Q[m]] = self.theta(self.df.iloc[-1,0], i, self.T, self.s, self.r, self.ot,j)
                            case 'vega':
                                Z[Q[p],Q[m]] = self.vega(self.df.iloc[-1,0], i, self.T, self.s, self.r, self.ot,j)
                        m = m+1
                    p = p+1
                return np.meshgrid(X,Y) , Z.transpose()
            case '23' | '32':
                if matching_indices == '32':
                    u = x_min
                    x_min = y_min
                    y_min = u
                    u = x_max
                    x_max = y_max
                    y_max = u
                    T = X
                    X = Y
                    Y = T
                p = 0
                for i in X:
                    m = 0
                    for j in Y:
                        match t:
                            case 'BS':
                                Z[Q[p],Q[m]] = self.BlackScholes(self.df.iloc[-1,0], self.K, i, j, self.r, self.ot,self.d)
                            case 'delta':
                                Z[Q[p],Q[m]] = self.delta(self.df.iloc[-1,0], self.K, i, j, self.r, self.ot,self.d)
                            case 'gamma':
                                Z[Q[p],Q[m]] = self.gamma(self.df.iloc[-1,0], self.K, i, j, self.r, self.ot,self.d)
                            case 'rho':
                                Z[Q[p],Q[m]] = self.rho(self.df.iloc[-1,0], self.K, i, j, self.r, self.ot,self.d)
                            case 'theta':
                                Z[Q[p],Q[m]] = self.theta(self.df.iloc[-1,0], self.K, i, j, self.r, self.ot,self.d)
                            case 'vega':
                                Z[Q[p],Q[m]] = self.vega(self.df.iloc[-1,0], self.K, i, j, self.r, self.ot,self.d)
                        m = m+1
                    p = p+1
                return np.meshgrid(X,Y) , Z.transpose()
            case '42' | '24':
                if matching_indices == '42':
                    u = x_min
                    x_min = y_min
                    y_min = u
                    u = x_max
                    x_max = y_max
                    y_max = u
                    T = X
                    X = Y
                    Y = T
                p = 0
                for i in X:
                    m = 0
                    for j in Y:
                        match t:
                            case 'BS':
                                Z[Q[p],Q[m]] = self.BlackScholes(self.df.iloc[-1,0], self.K, i,self.s,j, self.ot,self.d)
                            case 'delta':
                                Z[Q[p],Q[m]] = self.delta(self.df.iloc[-1,0], self.K, i,self.s,j, self.ot,self.d)
                            case 'gamma':
                                Z[Q[p],Q[m]] = self.gamma(self.df.iloc[-1,0], self.K, i,self.s,j, self.ot,self.d)
                            case 'rho':
                                Z[Q[p],Q[m]] = self.rho(self.df.iloc[-1,0], self.K, i,self.s,j, self.ot,self.d)
                            case 'theta':
                                Z[Q[p],Q[m]] = self.theta(self.df.iloc[-1,0], self.K, i,self.s,j, self.ot,self.d)
                            case 'vega':
                                Z[Q[p],Q[m]] = self.vega(self.df.iloc[-1,0], self.K, i,self.s,j, self.ot,self.d)
                        m = m+1
                    p = p+1
                return np.meshgrid(X,Y) , Z.transpose()
            case '25' | '52':
                if matching_indices == '52':
                    u = x_min
                    x_min = y_min
                    y_min = u
                    u = x_max
                    x_max = y_max
                    y_max = u
                    T = X
                    X = Y
                    Y = T
                p = 0
                for i in X:
                    m = 0
                    for j in Y:
                        match t:
                            case 'BS':
                                Z[Q[p],Q[m]] = self.BlackScholes(self.df.iloc[-1,0], self.K, i,self.s,self.r, self.ot,j)
                            case 'delta':
                                Z[Q[p],Q[m]] = self.delta(self.df.iloc[-1,0], self.K, i,self.s,self.r, self.ot,j)
                            case 'gamma':
                                Z[Q[p],Q[m]] = self.gamma(self.df.iloc[-1,0], self.K, i,self.s,self.r, self.ot,j)
                            case 'rho':
                                Z[Q[p],Q[m]] = self.rho(self.df.iloc[-1,0], self.K, i,self.s,self.r, self.ot,j)
                            case 'theta':
                                Z[Q[p],Q[m]] = self.theta(self.df.iloc[-1,0], self.K, i,self.s,self.r, self.ot,j)
                            case 'vega':
                                Z[Q[p],Q[m]] = self.vega(self.df.iloc[-1,0], self.K, i,self.s,self.r, self.ot,j)
                        m = m+1
                    p = p+1
                return np.meshgrid(X,Y) , Z.transpose()
            case '43' | '34':
                if matching_indices == '43':
                    u = x_min
                    x_min = y_min
                    y_min = u
                    u = x_max
                    x_max = y_max
                    y_max = u
                    T = X
                    X = Y
                    Y = T
                p = 0
                for i in X:
                    m = 0
                    for j in Y:
                        match t:
                            case 'BS':
                                Z[Q[p],Q[m]] = self.BlackScholes(self.df.iloc[-1,0], self.K, self.T,i,j, self.ot,self.d)
                            case 'delta':
                                Z[Q[p],Q[m]] = self.delta(self.df.iloc[-1,0], self.K, self.T,i,j, self.ot,self.d)
                            case 'gamma':
                                Z[Q[p],Q[m]] = self.gamma(self.df.iloc[-1,0], self.K, self.T,i,j, self.ot,self.d)
                            case 'rho':
                                Z[Q[p],Q[m]] = self.rho(self.df.iloc[-1,0], self.K, self.T,i,j, self.ot,self.d)
                            case 'theta':
                                Z[Q[p],Q[m]] = self.theta(self.df.iloc[-1,0], self.K, self.T,i,j, self.ot,self.d)
                            case 'vega':
                                Z[Q[p],Q[m]] = self.vega(self.df.iloc[-1,0], self.K, self.T,i,j, self.ot,self.d)
                        m = m+1
                    p = p+1
                return np.meshgrid(X,Y) , Z.transpose()
            case '35' | '53':
                if matching_indices == '53':
                    u = x_min
                    x_min = y_min
                    y_min = u
                    u = x_max
                    x_max = y_max
                    y_max = u
                    T = X
                    X = Y
                    Y = T
                p = 0
                for i in X:
                    m = 0
                    for j in Y:
                        match t:
                            case 'BS':
                                Z[Q[p],Q[m]] = self.BlackScholes(self.df.iloc[-1,0], self.K, self.T,i,self.r, self.ot,j)
                            case 'delta':
                                Z[Q[p],Q[m]] = self.delta(self.df.iloc[-1,0], self.K, self.T,i,self.r, self.ot,j)
                            case 'gamma':
                                Z[Q[p],Q[m]] = self.gamma(self.df.iloc[-1,0], self.K, self.T,i,self.r, self.ot,j)
                            case 'rho':
                                Z[Q[p],Q[m]] = self.rho(self.df.iloc[-1,0], self.K, self.T,i,self.r, self.ot,j)
                            case 'theta':
                                Z[Q[p],Q[m]] = self.theta(self.df.iloc[-1,0], self.K, self.T,i,self.r, self.ot,j)
                            case 'vega':
                                Z[Q[p],Q[m]] = self.vega(self.df.iloc[-1,0], self.K, self.T,i,self.r, self.ot,j)
                        m = m+1
                    p = p+1
                return np.meshgrid(X,Y) , Z.transpose()

            case '54' | '45':
                if matching_indices == '54':
                    u = x_min
                    x_min = y_min
                    y_min = u
                    u = x_max
                    x_max = y_max
                    y_max = u
                    T = X
                    X = Y
                    Y = T
                p = 0
                for i in X:
                    m = 0
                    for j in Y:
                        match t:
                            case 'BS':
                                Z[Q[p],Q[m]] = self.BlackScholes(self.df.iloc[-1,0], self.K, self.T,self.s,i, self.ot,j)
                            case 'delta':
                                Z[Q[p],Q[m]] = self.delta(self.df.iloc[-1,0], self.K, self.T,self.s,i, self.ot,j)
                            case 'gamma':
                                Z[Q[p],Q[m]] = self.gamma(self.df.iloc[-1,0], self.K, self.T,self.s,i, self.ot,j)
                            case 'rho':
                                Z[Q[p],Q[m]] = self.rho(self.df.iloc[-1,0], self.K, self.T,self.s,i, self.ot,j)
                            case 'theta':
                                Z[Q[p],Q[m]] = self.theta(self.df.iloc[-1,0], self.K, self.T,self.s,i, self.ot,j)
                            case 'vega':
                                Z[Q[p],Q[m]] = self.vega(self.df.iloc[-1,0], self.K, self.T,self.s,i, self.ot,j)
                        m = m+1
                    p = p+1
                return np.meshgrid(X,Y) , Z.transpose()
            case _:
                print("error")



