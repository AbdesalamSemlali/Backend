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



class scrapper:
    def __init__(self,pays : str,dstart : str):
        self.pays = pays
        # country code
        self.CC = None
        # date to upload data for
        self.dstart = dstart
        # url chosen based on pays variable to define it run the method URL
        self.url = None
        # the data corresponding to the date chosen to get it run scrape
        self.data = None
        # maturities gotten from the scrape
        self.MZC = None
    def __str__(self):
        return f'pays : {self.pays} \nDate de début : {self.dstart} \n'
    # check for country name if it's correct and gives the country code to scrape with it
    def code_C(self):
        country = CountryInfo(self.pays)
        if len(country.alt_spellings())==0:
            print("You didn't written the country name correctly")
        else:
             self.CC = country.alt_spellings()[0]
    # gives the appropriate url for each country code we have to urls one for morocco and one for the other contries
    def URL(self):
        self.code_C()
        if self.CC == 'MA':
            self.url = "https://www.bkam.ma/Marches/Principaux-indicateurs/Marche-obligataire/Marche-des-bons-de-tresor/Marche-secondaire/Taux-de-reference-des-bons-du-tresor?date="+self.dstart[0:2] + '%2F' + self.dstart[3:5] + '%2F' + self.dstart[6:] + "&block=e1d6b9bbf87f86f8ba53e8518e882982#address-c3367fcefc5f524397748201aee5dab8-e1d6b9bbf87f86f8ba53e8518e882982"
        elif self.CC != None:
            self.url = 'https://fr.tradingview.com/markets/bonds/prices-all'
    def scrape(self):
        self.URL()
        if self.CC == 'MA':
            import ssl 
            ssl._create_default_https_context = ssl._create_unverified_context
            page = requests.get(self.url ,verify=False)
            scrape = bs(page.text, 'lxml')
            t1 = scrape.find('table', {"class":"dynamic_contents_ref_12"})
            if t1 != None:
                df = pd.read_html(self.url, flavor='lxml')[0]
                df.dropna(inplace=True)
                df = df.iloc[:,[0,2,3]]
                df.iloc[:,1] = df.iloc[:,1].str.replace('\xa0%','').str.replace(',','.').astype('float') /100.0
                df.iloc[:,0] = (pd.to_datetime(df.iloc[:,0], format='%d/%m/%Y') - pd.to_datetime(df.iloc[:,-1], format='%d/%m/%Y')).dt.days
                df = df.iloc[:,[1,0]]
                df.columns = [self.CC + '_TBA','maturity']
                self.data = df
        elif self.CC != None:
            with sync_playwright() as p:
                browser = p.chromium.launch( executable_path="opt/render/.cache/ms-playwright/chromium-1091")
                page = browser.new_page()
                # Navigate to the webpage
                page.goto('https://fr.tradingview.com/markets/bonds/prices-all')
                # Click the button using a selector
                button_selector1 = '#overlap-manager-root > div > div.tv-dialog__modal-wrap > div > div > div > div.tv-dialog__close.close-Nc1uyYX2.dialog-close-Nc1uyYX2.js-dialog__close'
                page.click(button_selector1)
                page.wait_for_timeout(100)
                button_selector = 'button.button-SFwfC2e0'
                while page.query_selector(button_selector) is not None:
                    table = pd.read_html(StringIO(str(bs(str(page.content()), 'lxml').find('div', {"class":"tableWrap-SfGgNYTG"}))),flavor= 'lxml')[0]
                    if  table.empty : 
                        break
                    if any(table.iloc[:,0].str.contains(self.CC)):
                        page.click(button_selector)
                        page.wait_for_timeout(300)
                        table = pd.read_html(StringIO(str(bs(str(page.content()), 'lxml').find('div', {"class":"tableWrap-SfGgNYTG"}))),flavor= 'lxml')[0]
                        break
                    page.click(button_selector)
                    page.wait_for_timeout(300)
                table = pd.read_html(StringIO(str(bs(str(page.content()), 'lxml').find('div', {"class":"tableWrap-SfGgNYTG"}))),flavor= 'lxml')[0]
                browser.close()
                if table.empty :
                    print("This date doesn't contain any data")
                else:
                    table.dropna(inplace=True)
                    table =  table[table.iloc[:,0].str.contains(self.CC)]
                    self.MZC = table.iloc[:,0].str.split('Y',n = 1, expand = True).iloc[:,0] + 'Y'
                    self.MZC = self.MZC.to_frame().reset_index(drop=True)
                    table = table.iloc[:,[2,4]]
                    table.iloc[:,0] = table.iloc[:,0].str.replace('%','').str.replace('−','-').astype('float') /100.0
                    table.columns = [self.CC + '_TBA','maturity']
                    table.reset_index(drop=True,inplace=True)
                    table.iloc[:,1] = table.iloc[:,1].astype('int')
                    self.data = table
                
