from scipy.stats import norm
from Option.Classes import *
import matplotlib.pyplot as plt
import networkx as nx
class eOp(Op):
    def __init__(self,K : float,ticker : str,N : int,ot :str,exp : str,r : float,d : float,s : float):
        super().__init__(K,ticker,N,ot,exp,d,r,s)

    def CRR(self):
        dt = self.T / self.N
        u = np.exp(self.s * np.sqrt(dt))
        d = 1 / u
        q = (np.exp((self.r - self.d) * dt) - d) / (u - d)
        disc = np.exp(-self.r * dt)


        C=float(self.df.iloc[-1,0]) * d ** pd.Series(list(range(self.N, -1, -1))) * u **  pd.Series(list(range(0, self.N + 1, 1)))
        #C = float(self.df.iloc[-1]) * d ** np.arange(self.N, -1, -1) * u ** np.arange(0, self.N + 1, 1)
        if self.ot == "Call":
            C = pd.Series(np.maximum((C - self.K).tolist(), np.zeros(self.N + 1))).tolist()
        else:
            C = pd.Series(np.maximum((self.K - C).tolist(), np.zeros(self.N + 1))).tolist()


        for i in range(self.N, 0, -1):
            C = (disc * (q * pd.Series(C[1:i + 1]) + (1 - q) * pd.Series(C[0:i]))).tolist()
        return C[0]

    def TM(self):
        dt = self.T / self.N
        u = np.exp(self.s * np.sqrt(2 * dt))
        d = 1 / u
        a = np.exp(self.r * dt / 2)
        b = np.exp(self.s * np.sqrt(dt / 2))
        pu = ((a - 1/b) / (b-1/b)) ** 2
        p = ((b - a) / (b-1/b)) ** 2
        pm = 1 - pu - p
        disc = np.exp(-self.r * dt)


        C=float(self.df.iloc[-1,0]) * d ** pd.Series(list(np.arange(self.N, -1/2, -1/2))) * u **  pd.Series(list(np.arange(0, self.N + 1/2, 1/2)))
        #C = float(self.df.iloc[-1]) * d ** np.arange(self.N, -1/2, -1/2) * u ** np.arange(0, self.N + 1/2, 1/2)
        if self.ot == "Call":
            #C = np.maximum(C - self.K, np.zeros(2 * self.N + 1))
            C = pd.Series(np.maximum((C - self.K).tolist(), np.zeros(2*self.N + 1))).tolist()
        else:
            #C = np.maximum(self.K - C, np.zeros(2 * self.N + 1))
            C = pd.Series(np.maximum((self.K - C).tolist(), np.zeros(2*self.N + 1))).tolist()


        for i in range(2 * self.N, 0, -2):
            C = (disc * (pu * pd.Series(C[2:i + 1]) + (p * pd.Series(C[0:i-1])) + (pm * pd.Series(C[1:i])))).tolist()
            #C = disc * (pu * C[2:i + 1] + pm * C[1:i] + pd * C[0:i - 1])
        return C[0]

    def BS(self):
        d1 = (1 / (self.s * np.sqrt(self.T))) * (np.log(float(self.df.iloc[-1,0])/ self.K) + (self.r - self.d + 0.5 * (self.s) ** 2) * self.T)
        d2 = d1 - self.s * np.sqrt(self.T)


        if self.ot == "Call":
            return float(self.df.iloc[-1,0]) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.ot == "Put":
            return -float(self.df.iloc[-1,0]) * norm.cdf(-d1) + self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)

    def delta(self):
        d1 = (1 / (self.s * np.sqrt(self.T))) * (np.log(self.df.iloc[-1,0] / self.K) + ((self.r-self.d) + 0.5 * (self.s) ** 2) * self.T)
        return np.exp(-self.d*self.T)*norm.cdf(d1) if self.ot == "Call" else - np.exp(-self.d*self.T)*norm.cdf(-d1)

    def gamma(self):
        d1 = (1 / (self.s * np.sqrt(self.T))) * (np.log(self.df.iloc[-1,0] / self.K) + ((self.r-self.d)+ 0.5 * (self.s) ** 2) * self.T)
        return  np.exp(-self.d*self.T)* norm.pdf(d1) / (self.df.iloc[-1,0] * self.s * np.sqrt(self.T))

    def theta(self):
        d1 = (1 / (self.s * np.sqrt(self.T))) * (np.log(self.df.iloc[-1,0] / self.K) + ((self.r-self.d) + 0.5 * (self.s) ** 2) * self.T)
        d2 = d1 - self.s * np.sqrt(self.T)
        call = -np.exp(-self.d*self.T)*self.df.iloc[-1,0] * norm.pdf(d1, 0, 1) * self.s / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2, 0, 1) + self.d*self.df.iloc[-1,0]*np.exp(-self.d*self.T)*norm.cdf(d1,0,1)
        put = (-np.exp(-self.d*self.T)*self.df.iloc[-1,0] * self.s * norm.pdf(d1)) / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.d*self.df.iloc[-1,0]*np.exp(-self.d*self.T)*norm.cdf(-d1,0,1)
        return call  if self.ot == "Call" else put 

    def vega(self):
        d1 = (1 / (self.s * np.sqrt(self.T))) * (np.log(self.df.iloc[-1,0] / self.K) + ((self.r-self.d) + 0.5 * (self.s) ** 2) * self.T)
        d2 = d1 - self.s * np.sqrt(self.T)
        return np.exp(-self.d*self.T)*self.df.iloc[-1,0] * norm.pdf(d1) * np.sqrt(self.T)

    def rho(self):
        d1 = (1 / (self.s * np.sqrt(self.T))) * (np.log(self.df.iloc[-1,0] / self.K) + ((self.r-self.d) + 0.5 * (self.s) ** 2) * self.T)
        d2 = d1 - self.s * np.sqrt(self.T)
        call = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2, 0, 1)
        put = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2, 0, 1)
        return call  if self.ot == "Call" else put 


    def P_CRR(self,f):
        if self.N > 10:
            self.N = 10
        dt = self.T / self.N
        u = np.exp(self.s * np.sqrt(dt))
        d = 1 / u
        q = (np.exp((self.r - self.d) * dt) - d) / (u - d)
        disc = np.exp(-self.r * dt)

        C = float(self.df.iloc[-1,0]) * d ** pd.Series(list(range(self.N, -1, -1))) * u ** pd.Series(list(range(0, self.N + 1, 1)))
        T = []
        # C = float(self.df.iloc[-1]) * d ** np.arange(self.N, -1, -1) * u ** np.arange(0, self.N + 1, 1)
        if self.ot == "Call":
            C = pd.Series(np.maximum((C - self.K).tolist(), np.zeros(self.N + 1))).tolist()
            T.append(C)
        else:
            C = pd.Series(np.maximum((self.K - C).tolist(), np.zeros(self.N + 1))).tolist()
            T.append(C)

        for i in range(self.N, 0, -1):
            C = (disc * (q * pd.Series(C[1:i + 1]) + (1 - q) * pd.Series(C[0:i]))).tolist()
            T.append(C)

        G = nx.DiGraph()
        for i in range(self.N, -self.N - 1, -1):
            for j in range(i, -i - 1, -2):
                k = np.arange(0, i + 1, 1 / 2)
                G.add_node((i, j), label=f"{round(T[self.N - i][i - int(k[i - j])], 1)}")
                if i < self.N:
                    G.add_edge((i, j), (i + 1, j + 1), label=round(q, 3))
                    G.add_edge((i, j), (i + 1, j - 1), label=round(1 - q, 3))
        pos = {node: node for node in G.nodes()}
        labels = {node: G.nodes[node]["label"] for node in G.nodes()}
        plt.figure(figsize=(10, 10))
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=400, node_color='lightblue', font_size=8,
                font_color='black')
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        plt.title('Cox-Ross-Rubinstein tree for option pricing -Eur Case-')
        plt.axis('off')
        plt.savefig(f, format="png")

    def P_TM(self,f):
        if self.N > 10:
            self.N = 10
        dt = self.T / self.N
        u = np.exp(self.s * np.sqrt(2 * dt))
        d = 1 / u
        a = np.exp(self.r * dt / 2)
        b = np.exp(self.s * np.sqrt(dt / 2))
        pu = ((a - 1 / b) / (b - 1 / b)) ** 2
        p = ((b - a) / (b - 1 / b)) ** 2
        pm = 1 - pu - p
        disc = np.exp(-self.r * dt)
        C = float(self.df.iloc[-1,0]) * d ** pd.Series(list(np.arange(self.N, -1 / 2, -1 / 2))) * u ** pd.Series(list(np.arange(0, self.N + 1 / 2, 1 / 2)))
        T = []
        # C = float(self.df.iloc[-1]) * d ** np.arange(self.N, -1/2, -1/2) * u ** np.arange(0, self.N + 1/2, 1/2)
        if self.ot == "Call":
            # C = np.maximum(C - self.K, np.zeros(2 * self.N + 1))
            C = pd.Series(np.maximum((C - self.K).tolist(), np.zeros(2 * self.N + 1))).tolist()
        else:
            # C = np.maximum(self.K - C, np.zeros(2 * self.N + 1))
            C = pd.Series(np.maximum((self.K - C).tolist(), np.zeros(2 * self.N + 1))).tolist()
        T.append(C)
        for i in range(2 * self.N, 0, -2):
            C = (disc * (pu * pd.Series(C[2:i + 1]) + (p * pd.Series(C[0:i - 1])) + (pm * pd.Series(C[1:i])))).tolist()
            T.append(C)
            # C = disc * (pu * C[2:i + 1] + pm * C[1:i] + pd * C[0:i - 1])
        G = nx.DiGraph()
        for i in range(self.N, -self.N - 1, -1):
            for j in range(i, -i - 1, -1):
                k = np.arange(0, 2 * i + 1, 1)
                G.add_node((i, j), label=f"{round(T[self.N-i][int(k[i + j])], 1)}")
                if i < self.N:
                    G.add_edge((i, j), (i + 1, j + 1), label=round(pu, 3))
                    G.add_edge((i, j), (i + 1, j - 1), label=round(p, 3))
                    G.add_edge((i, j), (i + 1, j), label=round(pm, 3))
        pos = {node: node for node in G.nodes()}
        labels = {node: G.nodes[node]["label"] for node in G.nodes()}
        plt.figure(figsize=(10, 10))
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=400, node_color='lightblue', font_size=8,font_color='black')
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        plt.title('Boyle tree for option pricing -Eur Case-')
        plt.axis('off')
        plt.savefig(f, format="png")








