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
from Yield.Scrapper import *


class Interpolation(scrapper):
    def __init__(self,pays : str,dstart : str):
        super().__init__(pays,dstart) 
        self.code_C()
        self.URL()
        self.scrape()

    
    def bootsrapping(self):
        df = self.data.copy()
        for j in range(len(df.maturity)):
            if j< len(df.maturity)-1:
                a=[]
                for i in np.arange(df.iloc[j,-1]+1,df.iloc[j+1,-1],5):
                    a.append(((df.iloc[j+1,-1]-i)*df.iloc[j,0]+(i-df.iloc[j,-1])*df.iloc[j+1,0])/(df.iloc[j+1,-1]-df.iloc[j,-1]))
                df1 = pd.DataFrame(np.arange(df.iloc[j,-1]+1,df.iloc[j+1,-1],5),columns=['maturity'])
                df1[self.CC + '_TBA'] = a
                df1 = df1.iloc[:,[1,0]]
                df = pd.concat([df, df1])
        #        
        self.BS = df.drop_duplicates().sort_values(by='maturity',ignore_index=True)
        #

    def LI(self):
        df = self.data.copy()
        p = df['maturity']/360
        m = np.array([np.ones(len(p)),p]).astype(float)
        mt = m.transpose()
        a = np.linalg.inv(np.dot(m,mt))
        a = np.dot(np.dot(a,m),np.array(df.iloc[:,0]))
        df1 = pd.DataFrame(np.arange(df.iloc[0,-1]+1,df.iloc[-1,-1],5),columns=['maturity'])
        p = df1['maturity']/360
        m = np.array([np.ones(len(p)),p])
        df1[self.CC + '_TBA'] = np.dot(a,m)
        df1 = df1.iloc[:,[1,0]]
        #
        self.li = df1.sort_values(by='maturity',ignore_index=True)
        #
    
    def CI(self):
        df = self.data.copy()
        p = df['maturity']/360
        m = np.array([np.ones(len(p)),p,(p)**2,(p)**3]).astype(float)
        mt = m.transpose()
        a = np.linalg.inv(np.dot(m,mt))
        a = np.dot(np.dot(a,m),np.array(df.iloc[:,0]))
        df1 = pd.DataFrame(np.arange(df.iloc[0,-1]+1,df.iloc[-1,-1],5),columns=['maturity'])
        p = df1['maturity']/360
        m = np.array([np.ones(len(p)),p,(p)**2,(p)**3])
        df1[self.CC + '_TBA'] = np.dot(a,m)
        df1 = df1.iloc[:,[1,0]]
        #
        self.ci = df1.sort_values(by='maturity',ignore_index=True)
        #

    
    def NSF(self,x:list):
        df1 = self.data.copy()
        p = (df1['maturity']/360).astype(float)
        s = (1-np.exp(-(p*x[-1])))/(p*x[-1])
        m = np.array([np.ones(len(p)),s,(s-np.exp(-p*x[-1]))])
        a = np.dot(x[0:-1],m)
        return sum((a-df1.iloc[:,0])**2)

    def NS(self):
        x = [0.1,0.01,0.01,1]
        res = minimize(self.NSF, x, method='BFGS', tol=1e-10)
        df = self.data.copy()
        df1 = pd.DataFrame(np.arange(df.iloc[0,-1]+1,df.iloc[-1,-1],5),columns=['maturity'])
        p = df1['maturity']/360
        s = (1-np.exp(-res.x[-1]*p))/(res.x[-1]*p)
        m = np.array([np.ones(len(p)),s,s-np.exp(-res.x[-1]*p)])
        df1[self.CC+'_TBA'] = np.dot(res.x[0:-1],m)
        df1 = df1.iloc[:,[1,0]]
        #
        self.ns = df1.sort_values(by='maturity',ignore_index=True)
        #

    def NSSF(self,x:list):
        df1 = self.data.copy()
        p = (df1['maturity']/360).astype(float)
        s1 = (1-np.exp(-(p*x[-1])))/(p*x[-1])
        s = (1-np.exp(-(p*x[-2])))/(p*x[-2])
        m = np.array([np.ones(len(p)),s,(s-np.exp(-p*x[-2])),(s1-np.exp(-p*x[-1]))])
        a = np.dot(x[0:-2],m)
        return sum((a-df1.iloc[:,0])**2)
    
    def NSS(self):
        x = [0.1,0.01,0.01,0.01,1,2]
        res = minimize(self.NSSF, x,method='BFGS', tol=1e-10)
        df = self.data.copy()
        df1 = pd.DataFrame(np.arange(df.iloc[0,-1]+1,df.iloc[-1,-1],5),columns=['maturity'])
        p = df1['maturity']/360
        s1 = (1-np.exp(-(p*res.x[-1])))/(p*res.x[-1])
        s = (1-np.exp(-(p*res.x[-2])))/(p*res.x[-2])
        m = np.array([np.ones(len(p)),s,(s-np.exp(-p*res.x[-2])),(s1-np.exp(-p*res.x[-1]))])
        df1[self.CC+'_TBA'] = np.dot(res.x[0:-2],m) 
        df1 = df1.iloc[:,[1,0]] 
        #
        self.nss = df1.sort_values(by='maturity',ignore_index=True)
        #
    
    def SC(self,n:int):
        df1 = self.data.copy()
        p = (df1['maturity']/360).astype(float)
        if n>len(p):
            n = len(p)-1

        x=[]
        y=[]
        for i in range(n):
            x.append(p.iloc[int(len(p)/n)*(i)])
            y.append(df1.iloc[int(len(p)/n)*(i),0])
        x.append(p.iloc[-1])
        x = pd.DataFrame(x)
        
        y.append(df1.iloc[-1,0])
        y = pd.DataFrame(y)



        h = (x - x.shift(1)).dropna()
        f = (6*((y.shift(-2) - y.shift(-1)) / ((x.shift(-2)-x.shift(-1))*(x.shift(-2)-x))) - ((y.shift(-1) - y) /((x.shift(-1)-x)*(x.shift(-2)-x)))).dropna()
        f = pd.concat([pd.DataFrame([6*(y.iloc[1]-y.iloc[0])/(x.iloc[1]-x.iloc[0])**2]),f,pd.DataFrame([6*(y.iloc[-2]-y.iloc[-1])/(x.iloc[-1]-x.iloc[-2])**2])],ignore_index=True).dropna()
        u = (h/(h+h.shift(-1))).dropna()
        l = 1- u
        u = pd.concat([u,pd.DataFrame([1])],ignore_index=True)
        l = pd.concat([pd.DataFrame([1]),l],ignore_index=True).dropna()



        r = np.zeros((n+1,n+1))
        np.fill_diagonal(r,2)
        np.fill_diagonal(r[:-1,1:],l)
        np.fill_diagonal(r[1:,:-1],u)



        f = np.array(f)
        M = np.dot(np.linalg.inv(r),f)
        M = pd.DataFrame(M)


        A = np.array((M.shift(1)/(6*h)).dropna())
        B = np.array((M/(6*h)).dropna())
        C = np.array(((y.shift(1)-((M.shift(1)*h**2)/6))/(h)).dropna())
        D = np.array(((y-((M*h**2)/6))/h).dropna())
        


        df2 = pd.DataFrame(np.arange(df1.iloc[0,-1]+1,df1.iloc[-1,-1],5),columns=['maturity'])
        df2['maturity'] = df2['maturity']/360
        z = np.array([])
        Z = np.array([])
        
        for i in range (n-1):
            if int(len(p)/n)*(i+1)<len(p):
                z1 = np.array(df2.loc[(df2['maturity']>p.iloc[int(len(p)/n)*(i)]) & (df2['maturity']<p.iloc[int(len(p)/n)*(i+1)])]).flatten()
                Z = np.append(Z,z1)
                z = np.append(z,A[i]*(p.iloc[int(len(p)/n)*(i+1)]-z1)**3+B[i]*(z1-p.iloc[int(len(p)/n)*(i)])**3+C[i]*(p.iloc[int(len(p)/n)*(i+1)]-z1)+D[i]*(z1-p.iloc[int(len(p)/n)*(i)]))
        i = n-1
        z1 = np.array(df2.loc[(df2['maturity']>p.iloc[int(len(p)/n)*(i)]) & (df2['maturity']<p.iloc[-1])]).flatten()
        Z = np.append(Z,z1)
        z = np.append(z,A[i]*(p.iloc[-1]-z1)**3+B[i]*(z1-p.iloc[int(len(p)/n)*(i)])**3+C[i]*(p.iloc[-1]-z1)+D[i]*(z1-p.iloc[int(len(p)/n)*(i)]))
        ds = pd.DataFrame(Z*360,columns=['maturity'])
        ds[self.CC+'_TBA'] = z
        ds = ds.iloc[:,[1,0]]
        #
        self.sc = ds.sort_values(by='maturity',ignore_index=True)
        #

    def SC1(self,t:list):
        
        df1 = self.data.copy()
        p = (df1['maturity']/360).astype(float)
        

        x=np.array([])
        y=np.array([])
        x = np.append(x,p.iloc[0])
        y = np.append(y,df1.iloc[0,0])
        for i in range(len(t)):
            x = np.append(x,p.loc[p<=t[i]].iloc[-1])
            y = np.append(y,df1.loc[df1['maturity']<=360*t[i]].iloc[-1,0])
        x = np.append(x,p.iloc[-1])
        y = np.append(y,df1.iloc[-1,0])
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        x.drop_duplicates(inplace=True,ignore_index=True)
        y.drop_duplicates(inplace=True,ignore_index=True)



        h = (x - x.shift(1)).dropna().reset_index(drop=True)
        f1 = (y.shift(-2) - y.shift(-1)).dropna().reset_index(drop=True) / ((x.shift(-2)-x.shift(-1)).dropna().reset_index(drop=True)*(x.shift(-2)-x).dropna().reset_index(drop=True))
        f2 =  ((y.shift(-1) - y).dropna().reset_index(drop=True) /((x.shift(-1)-x).dropna().reset_index(drop=True)*(x.shift(-2)).dropna().reset_index(drop=True)))
        f = 6*(f1.dropna().reset_index(drop=True) - f2.dropna().reset_index(drop=True))
        f = pd.concat([pd.DataFrame([6*(y.iloc[1]-y.iloc[0])/(x.iloc[1]-x.iloc[0])**2]),f,pd.DataFrame([6*(y.iloc[-2]-y.iloc[-1])/(x.iloc[-1]-x.iloc[-2])**2])],ignore_index=True).dropna().reset_index(drop=True)
        u = (h/(h+h.shift(-1))).dropna().reset_index(drop=True)
        l = 1- u
        u = pd.concat([u,pd.DataFrame([1])],ignore_index=True).reset_index(drop=True)
        l = pd.concat([pd.DataFrame([1]),l],ignore_index=True).dropna().reset_index(drop=True)



        r = np.zeros((len(x),len(x)))
        np.fill_diagonal(r,2)
        np.fill_diagonal(r[:-1,1:],l)
        np.fill_diagonal(r[1:,:-1],u)

        

        f = np.array(f)
        M = np.dot(np.linalg.inv(r),f)
        M = pd.DataFrame(M)
        A = np.array((M.shift(1).dropna().reset_index(drop=True)/(6*h)).dropna())
        B = np.array((M/(6*h)).dropna())
        C = np.array(((y.shift(1).dropna().reset_index(drop=True)-((M.shift(1).dropna().reset_index(drop=True)*h**2)/6))/(h))) 
        D = np.array(((y-((M*h**2)/6))/h))



        df2 = pd.DataFrame(np.arange(df1.iloc[0,-1]+1,df1.iloc[-1,-1],5),columns=['maturity'])
        df2['maturity'] = df2['maturity']/360
        z = np.array([])
        Z = np.array([])
        for i in range (len(x.iloc[1:-1])):
            z1 = np.array(df2.loc[(df2['maturity']>x.iloc[i,0]) & (df2['maturity']<x.iloc[i+1,0])]).flatten()
            Z = np.append(Z,z1)
            z = np.append(z,A[i]*(x.iloc[i+1,0]-z1)**3+B[i]*(z1-x.iloc[i,0])**3+C[i]*(x.iloc[i+1,0]-z1)+D[i]*(z1-x.iloc[i,0]))
        i = len(x.iloc[1:-1])
        z1 = np.array(df2.loc[(df2['maturity']>x.iloc[-2,0]) & (df2['maturity']<x.iloc[-1,0])]).flatten()
        Z = np.append(Z,z1)
        z = np.append(z,A[i]*(x.iloc[-1,0]-z1)**3+B[i]*(z1-x.iloc[-2,0])**3+C[i]*(x.iloc[-1,0]-z1)+D[i]*(z1-x.iloc[-2,0]))
        ds = pd.DataFrame(Z*360,columns=['maturity'])
        ds[self.CC+'_TBA'] = z
        ds = ds.iloc[:,[1,0]]
        #
        self.sc1 = ds.sort_values(by='maturity',ignore_index=True)
        #



