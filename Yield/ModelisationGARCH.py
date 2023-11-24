from countryinfo import CountryInfo
import requests 
from bs4 import BeautifulSoup as bs
import numpy as np
import re
import pandas as pd
from playwright.sync_api import sync_playwright
from multiprocessing import Pool,cpu_count,freeze_support
from datetime import datetime,timedelta
import time
from io import StringIO
from tvDatafeed import TvDatafeed, Interval
from scipy.optimize import minimize,LinearConstraint
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
from Yield.BDscrapper import BDscrapper


class ModelisationGARCH(BDscrapper):
    def __init__(self, pays: str, dstart: str, dend: str):
        super().__init__(pays, dstart, dend)
        self.code_C()
        self.MPBD()
        


    def VASMLE(self,B):
        df = self.data.iloc[:,-1]
        Y = np.array( B[0]*B[1]+ B[0]*np.array(df.shift(1).dropna()))
        YR = np.array((df.shift(-1)-df).dropna())
        return sum((YR-Y)**2)
    def VASI(self):
        B = [0.1,0.1]
        res = minimize(self.VASMLE,B,method='BFGS',tol=1e-100)
        res.x = np.absolute(res.x)                   
        self.PV =  res.x
    def VGARCHMLE(self,C):
        B = self.PV
        df = self.data.iloc[:,-1]
        Y = np.array(df.shift(1).dropna()) + B[0]*B[1] - B[0]*np.array(df.shift(1).dropna())
        YR = np.array((df.shift(-1)).dropna())
        x = np.abs(YR-Y)
        l = np.zeros(len(df)-1)
        l[0] = (C[0]/(1-C[1]-C[2]))**(1/2)
        for i in range(1,len(self.data.index)-1):
            l[i]=(C[0]+C[1]*x[i-1]**2+C[2]*l[-1]**2)**(1/2)
        L = 1/((2*math.pi)**(1/2)*l) * np.exp(-x**2/(2*l**2)) 
        MLE = sum(np.log(L)) 
        return -MLE
    def VGARCH(self):
        self.VASI()
        df = self.data.iloc[:,-1]
        s = float(df.std())
        C = [s,0,0]
        res = minimize(self.VGARCHMLE, C,bounds=[(0, None), (0, None), (0, None)],constraints=({'type': 'eq', 'fun': lambda C: C[0]/(1-C[1]-C[2])}))
        self.PV1 = res.x
    def VASICEKG(self):
        self.VGARCH()
        B = self.PV
        df = self.data.iloc[:,-1]
        Y = np.array(df.shift(1).dropna()) + B[0]*B[1] - B[0]*np.array(df.shift(1).dropna())
        YR = np.array((df.shift(-1)).dropna())
        C = self.PV1
        l = np.zeros(len(df)-1)
        l[0] = (C[0]/(1-C[1]-C[2]))**(1/2)
        k = [df.iloc[0]]
        x = np.abs(YR-Y)
        for i in range(1,len(self.data.index)-1):
            W = np.random.normal(0,1,size=len(self.data.index)-1)
            l[i]=(C[0]+C[1]*x[i-1]**2+C[2]*l[-1]**2)**(1/2)
            k.append(k[-1]*np.exp(-B[0])+B[1]*(1-np.exp(-B[0]))+l[i]*W[i])
        k = pd.DataFrame(k,index=self.data.index[:-1])
        return k 

    def VASICEKGP(self):
        B = self.PV
        C = self.PV1
        Y = (np.array(self.data.iloc[-2,-1])+B[0]*B[1] - B[0]*np.array(self.data.iloc[-2,-1]))
        YR = np.array(self.data.iloc[-1,-1])
        l = [np.abs(YR-Y)]
        k = [self.data.iloc[-2,-1]]
        for i in range(360):
            W = np.random.normal(0,1,size=360)
            k.append(k[-1]+self.PV[0]*(self.PV[1]-k[-1])+l[-1]*W[i])
            l.append((C[0]+C[1]*(k[-1]-k[-2])**2+C[2]*l[-1]**2)**(1/2))
        k = pd.DataFrame(k,index=np.array(pd.date_range(start= self.data.index[-2], end=(pd.to_datetime(self.data.index[-2]) + pd.to_timedelta(360, unit='D')), freq='D')))
        return k
    def VASICEKGPMCS(self,n):
        df = self.VASICEKGP()
        for i in range(n):
            df = pd.concat([df,self.VASICEKGP()],axis=1)
        return df
    def VASICEKGPM(self):
        df = self.VASICEKGPMCS(3000)
        df1 = df.mean(axis=1)
        return df1


    



"""
b = 'MA'            
d = ModelisationGARCH(b, '01-01-2013','01-01-2023')
#f = ModelisationAR(b, '01-01-2013','01-01-2023')
l= np.array(d.CIR())
l1 = np.array(d.VASICEK())
#L,a = f.VASICEK('MLE')
#L1,a1 = f.CIR('MLE')
K = np.array(l+l1)/2#np.array(L[1:])+np.array(L1[1:])+


x = np.arange(0,len(d.data.index))
#plt.plot(x[1:],K,c = 'pink',label='mean of the four '+d.CC)
#plt.plot(x[1:],l,c = 'green',label='CIR GARCH '+d.CC)
plt.plot(x[1:],l1,c = 'yellow',label='VASICEK GARCH '+d.CC)
#plt.plot(x,L,c = 'red',label='Vasicek MLE '+d.CC)
#plt.plot(x,L1,c = 'purple',label='CIR MLE '+d.CC)
plt.plot(x,d.data.iloc[:,-1],c = 'black',label='Real Data of '+d.CC)
plt.title('modeling short interest rate for '+d.CC)
plt.legend()
plt.tick_params(axis='x',which='both', bottom=False,  top=False, labelbottom=False)
plt.xticks(ticks=None,rotation=45)
plt.show()
"""
