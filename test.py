from Classes import *
from euoption import *

newOption = eOp(K=100,ticker="AAPL",N=10,ot="Call",exp="2023-10-31")
newOption.D()
newOption.volatility()
price = newOption.CRR()
St = newOption.df.iloc[-1].values[0]
s=newOption.s

print(s)