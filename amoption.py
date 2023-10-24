from Classes import *
import matplotlib.pyplot as plt
import networkx as nx
class aOp(Op):
    def __init__(self,K : float,ticker : str,N : int,ot :str,r=0.05,d=0.0,s=0.0):
        super().__init__(K,ticker,N,ot,r,d,s)

    def CRR(self):
        dt = self.T / self.N
        u = np.exp(self.s * np.sqrt(dt))
        d = 1 / u
        q = (np.exp((self.r - self.d) * dt) - d) / (u - d)
        disc = np.exp(-self.r * dt)


        S=float(self.df.iloc[-1]) * d ** pd.Series(list(range(self.N, -1, -1))) * u **  pd.Series(list(range(0, self.N + 1, 1)))
        #S = St * d ** (np.arange(N, -1, -1)) * u ** (np.arange(0, N + 1, 1))
        if self.ot == 'Call':
            C = pd.Series(np.maximum((S - self.K).tolist(), np.zeros(self.N + 1))).tolist()
        else:
            C = pd.Series(np.maximum((self.K - S).tolist(), np.zeros(self.N + 1))).tolist()

        for i in range(self.N - 1, -1, -1):
            S = float(self.df.iloc[-1]) * d ** pd.Series(list(range(i, -1, -1))) * u ** pd.Series(list(range(0, i + 1, 1)))
            #S = St * d ** (np.arange(i, -1, -1)) * u ** (np.arange(0, i + 1, 1))
            #C[:i + 1] = disc * (q * C[1:i + 2] + (1 - q) * C[0:i + 1])
            C[:i+1] = (disc * (q * pd.Series(C[1:i + 2]) + (1 - q) * pd.Series(C[0:i+1]))).tolist()
            C = C[:-1]
            if self.ot == 'Call':
                C = pd.Series(np.maximum((S - self.K).tolist(), C)).tolist()
            else:
                C = pd.Series(np.maximum((self.K - S).tolist(), C)).tolist()
        return C[0]

    def TM(self):
        dt = self.T / self.N
        u = np.exp(self.s * np.sqrt(2 * dt))
        d = 1 / u
        a = np.exp(self.r * dt / 2)
        b = np.exp(self.s * np.sqrt(dt / 2))
        pu = ((a - 1 / b) / (b - 1 / b)) ** 2
        p = ((b - a) / (b - 1 / b)) ** 2
        pm = 1 - pu - p
        disc = np.exp(-self.r * dt)

        C = float(self.df.iloc[-1]) * d ** pd.Series(list(np.arange(self.N, -1 / 2, -1 / 2))) * u ** pd.Series(list(np.arange(0, self.N + 1 / 2, 1 / 2)))
        if self.ot == 'Call':
            C = pd.Series(np.maximum((C - self.K).tolist(), np.zeros(2*self.N + 1))).tolist()
        else:
            C = pd.Series(np.maximum((self.K - C).tolist(), np.zeros(2*self.N + 1))).tolist()
        for i in range(self.N-1, -1, -1):
            S = float(self.df.iloc[-1]) * d ** pd.Series(list(np.arange(i, -1/2, -1/2))) * u ** pd.Series(list(np.arange(0, i + 1/2, 1/2)))
            C[:2*i + 1] = (disc * (pu * pd.Series(C[0:2*i+1]) + (p * pd.Series(C[2:2*i + 3])) + (pm * pd.Series(C[1:2*i+2])))).tolist()
            C = C[:2*i + 1]
            if self.ot == 'Call':
                C = pd.Series(np.maximum((S - self.K).tolist(), C)).tolist()
            else:
                C = pd.Series(np.maximum((self.K - S).tolist(), C)).tolist()
        return C[0]

    def P_CRR(self):
        if self.N>10:
            self.N = 10
        dt = self.T / self.N
        u = np.exp(self.s * np.sqrt(dt))
        d = 1 / u
        q = (np.exp((self.r - self.d) * dt) - d) / (u - d)
        disc = np.exp(-self.r * dt)

        S = float(self.df.iloc[-1]) * d ** pd.Series(list(range(self.N, -1, -1))) * u ** pd.Series(
            list(range(0, self.N + 1, 1)))
        # S = St * d ** (np.arange(N, -1, -1)) * u ** (np.arange(0, N + 1, 1))
        T = []
        D = []
        if self.ot == 'Call':
            C = pd.Series(np.maximum((S - self.K).tolist(), np.zeros(self.N + 1))).tolist()
            T.append(C)
            D.append(C)
        else:
            C = pd.Series(np.maximum((self.K - S).tolist(), np.zeros(self.N + 1))).tolist()
            T.append(C)
            D.append(C)

        for i in range(self.N - 1, -1, -1):
            S = float(self.df.iloc[-1]) * d ** pd.Series(list(range(i, -1, -1))) * u ** pd.Series(list(range(0, i + 1, 1)))
            # S = St * d ** (np.arange(i, -1, -1)) * u ** (np.arange(0, i + 1, 1))
            # C[:i + 1] = disc * (q * C[1:i + 2] + (1 - q) * C[0:i + 1])
            C[:i + 1] = (disc * (q * pd.Series(C[1:i + 2]) + (1 - q) * pd.Series(C[0:i + 1]))).tolist()
            C = C[:-1]
            if self.ot == 'Call':
                D.append((S - self.K).tolist())
                T.append(C)
                C = pd.Series(np.maximum((S - self.K).tolist(), C)).tolist()
            else:
                D.append((self.K - S).tolist())
                T.append(C)
                C = pd.Series(np.maximum((self.K - S).tolist(), C)).tolist()
        G = nx.DiGraph()
        for i in range(self.N, -self.N - 1, -1):
            for j in range(i, -i - 1, -2):
                k = np.arange(0, i + 1, 1 / 2)
                if T[self.N - i][int(k[i + j])]<D[self.N - i][int(k[i + j])]:
                    A = "exercé"
                else:
                    A = ""
                G.add_node((i, j), label=f"{round(T[self.N - i][int(k[i + j])], 3)}\n{round(D[self.N - i][int(k[i + j])], 3)}\n\n{A}")
                if i < self.N:
                    G.add_edge((i, j), (i + 1, j + 1), label=round(q, 2))
                    G.add_edge((i, j), (i + 1, j - 1), label=round(1 - q, 2))
        pos = {node: node for node in G.nodes()}
        labels = {node: G.nodes[node]["label"] for node in G.nodes()}
        plt.figure(figsize=(10, 10))
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=600, node_color='lightblue', font_size=8,font_color='black')
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        plt.title('Cox-Ross-Rubinstein tree for option pricing -Amr Case-')
        plt.axis('off')

    def P_TM(self):
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

        C = float(self.df.iloc[-1]) * d ** pd.Series(list(np.arange(self.N, -1 / 2, -1 / 2))) * u ** pd.Series(list(np.arange(0, self.N + 1 / 2, 1 / 2)))
        if self.ot == 'Call':
            C = pd.Series(np.maximum((C - self.K).tolist(), np.zeros(2 * self.N + 1))).tolist()
        else:
            C = pd.Series(np.maximum((self.K - C).tolist(), np.zeros(2 * self.N + 1))).tolist()
        T = []
        D = []
        T.append(C)
        D.append(C)
        for i in range(self.N - 1, -1, -1):
            S = float(self.df.iloc[-1]) * d ** pd.Series(list(np.arange(i, -1 / 2, -1 / 2))) * u ** pd.Series(list(np.arange(0, i + 1 / 2, 1 / 2)))
            C[:2 * i + 1] = (disc * (pu * pd.Series(C[0:2 * i + 1]) + (p * pd.Series(C[2:2 * i + 3])) + (pm * pd.Series(C[1:2 * i + 2])))).tolist()
            C = C[:2 * i + 1]
            if self.ot == 'Call':
                T.append(C)
                D.append((S - self.K).tolist())
                C = pd.Series(np.maximum((S - self.K).tolist(), C)).tolist()
            else:
                T.append(C)
                D.append((S - self.K).tolist())
                C = pd.Series(np.maximum((self.K - S).tolist(), C)).tolist()
        G = nx.DiGraph()
        for i in range(self.N, -self.N - 1, -1):
            for j in range(i, -i - 1, -1):
                k = np.arange(0, 2 * i + 1, 1)
                if T[self.N - i][int(k[i + j])] < D[self.N - i][int(k[i + j])]:
                    A = "exercé"
                else:
                    A = ""
                G.add_node((i, j),
                           label=f"{round(T[self.N - i][int(k[i + j])], 3)}\n{round(D[self.N - i][int(k[i + j])], 3)}\n\n{A}")
                if i < self.N:
                    G.add_edge((i, j), (i + 1, j + 1), label=round(pu, 3))
                    G.add_edge((i, j), (i + 1, j - 1), label=round(p, 3))
                    G.add_edge((i, j), (i + 1, j), label=round(pm, 3))
        pos = {node: node for node in G.nodes()}
        labels = {node: G.nodes[node]["label"] for node in G.nodes()}
        plt.figure(figsize=(10, 10))
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=600, node_color='lightblue', font_size=8,
                font_color='black')
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        plt.title('Cox-Ross-Rubinstein tree for option pricing -Amr Case-')
        plt.axis('off')
