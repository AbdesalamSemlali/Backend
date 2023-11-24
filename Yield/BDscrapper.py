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
from Yield.Scrapper import scrapper




class BDscrapper(scrapper):
    def __init__(self,pays : str,dstart : str,dend : str):
        scrapper.__init__(self,pays,dstart)
        # country code initialize using super class
        self.code_C()
        # date to upload data for
        self.dend = dend

        scrapper.URL(self)
        scrapper.scrape(self)
             
        #
        self.urlBD = None
        #
        self.date_list = None
    
    def URL(self):
        if self.CC == 'MA':
            bkm = "https://www.bkam.ma/Marches/Principaux-indicateurs/Marche-obligataire/Marche-des-bons-de-tresor/Marche-secondaire/Taux-de-reference-des-bons-du-tresor?date="
            start = datetime.strptime(self.dstart, "%d-%m-%Y")
            end = datetime.strptime(self.dend, "%d-%m-%Y")
            date_list = pd.date_range(start, end, freq='D')
            x = pd.DataFrame(date_list,columns=['Date'])
            date_list = date_list.strftime("%d-%m-%Y")
            url = list()
            for i in date_list:
                i = i[0:2] + '%2F' + i[3:5] + '%2F' + i[6:]
                url.append(bkm + i)
            self.urlBD = url
            self.date_list = x


    def scrape(self,url1):
        if self.CC == 'MA':
            page = requests.get(url1)
            scrape = bs(page.text, 'lxml')
            t1 = scrape.find('table', {"class":"dynamic_contents_ref_12"})
            if t1 != None :
                df = pd.read_html(StringIO(str(t1)), flavor='lxml')[0].head(1)
                df = df.iloc[:,[0,2,3]]
                df.iloc[:,1] = df.iloc[:,1].str.replace('\xa0%','').str.replace(',','.').astype('float') /100.0
                df.iloc[:,0] = (pd.to_datetime(df.iloc[:,0], format='%d/%m/%Y') - pd.to_datetime(df.iloc[:,-1], format='%d/%m/%Y')).dt.days
                df.iloc[:,-1] = pd.to_datetime(df.iloc[:,-1], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
                df = df.iloc[:,[1,0,2]]
                df.columns = [self.CC + '_TBA','maturity','Date']
                self.data = pd.read_csv(self.CC + "_BD.csv")
                if all(self.data.iloc[:,-1] != df.iloc[0,-1]):
                    print('s')
                    df.to_csv(self.CC + "_BD.csv",mode='a',header=False, index=False)
                else:
                    return
        elif self.CC != None:
            tv = TvDatafeed()
            #for i in self.MZC:
            symbol = tv.search_symbol(self.MZC.iloc[0,0])[0]['symbol']
            exchange = tv.search_symbol(self.MZC.iloc[0,0])[0]['exchange']
            end = datetime.strptime(self.dend, '%d-%m-%Y')
            start = datetime.strptime(self.dstart, '%d-%m-%Y')
            self.data = tv.get_hist(symbol=symbol,exchange=exchange,interval=Interval.in_daily,n_bars=(end-start).days+200)
            self.data = self.data[['symbol','close']]
            self.data['symbol']= [self.MZC.iloc[0,0][2:5]]*len(self.data['symbol'])
            self.data.rename(columns={'symbol':'maturity'},inplace=True)
            s = self.MZC.iloc[0,0][:2] + '_TBA'
            self.data.rename(columns={'close':s},inplace=True)
            self.data.rename_axis(None,inplace=True)
            self.data = self.data[(self.data.index>=start) & (self.data.index<=end)]
            self.data.index = self.data.index.strftime('%Y-%m-%d')
            self.data.iloc[:,1]= self.data.iloc[:,1].astype('float') /100.0
            
            
    def MPBD(self):
            if self.CC == 'MA':
                freeze_support()
                self.URL()
                self.data = pd.read_csv(self.CC + "_BD.csv")
                self.data['Date'] = pd.to_datetime(self.data['Date'], format="%Y-%m-%d")
                self.data.drop_duplicates(inplace=True,ignore_index=True)
                self.data.sort_values(by='Date')
                url1 = np.array(self.urlBD)[self.date_list[(self.date_list<self.data['Date'].iloc[-1])].dropna().index.tolist()].tolist() + np.array(self.urlBD)[self.date_list[(self.date_list>self.data['Date'].iloc[0])].dropna().index.tolist()].tolist()
                num_processes =  cpu_count()
                if __name__ == '__main__':
                    with Pool(num_processes-3) as pool: 
                        pool.map(self.scrape, url1)
                        pool.close()
                        pool.terminate()
                        pool.join()
                self.data = pd.read_csv(self.CC + "_BD.csv")
                self.data['Date'] = pd.to_datetime(self.data['Date'], format="%Y-%m-%d")
                self.data.drop_duplicates(inplace=True,ignore_index=True)
                self.data.set_index('Date',drop=True,inplace=True)
                end = datetime.strptime(self.dend, '%d-%m-%Y')
                start = datetime.strptime(self.dstart, '%d-%m-%Y')
                self.data.index = pd.to_datetime(self.data.index, format='%d-%m-%Y')
                self.data = self.data[(self.data.index>=start) & (self.data.index<=end)]
                self.data.sort_index(inplace=True)
                self.data = self.data.iloc[:,[1,0]]
            else:
                self.scrape(self.url)

