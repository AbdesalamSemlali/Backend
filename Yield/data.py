from bs4 import BeautifulSoup
import requests
import re
import pandas as pd 

def scrapData(from_year,to_year) : 
    yield_data = []
    for i in range(from_year,to_year+1):
        url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value={i}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")


        table = soup.find("table")
        rows = table.find_all("tr")

        yield_data+=list(map(lambda x:re.sub(r' +',"",x.text.replace('N/A','')),rows))
        df=pd.DataFrame([sub.split('\n') for sub in yield_data]).drop_duplicates()                                  
        new_header = df.iloc[0]                                                                                      
        df = df[1:]                                                                                                  
        df.columns = new_header                                                                                      
        df['Date']=pd.to_datetime(df['Date'])                                                                        
        df.set_index('Date', inplace=True)                                                                           
        df=df.replace('',pd.NA).dropna(how='all', axis=1)                                                           
        df.columns = ["1 Mo",'2 Mo','3 Mo','4 Mo','6 Mo','1 Yr','2 Yr','3 Yr','5 Yr','7 Yr','10 Yr','20 Yr','30 Yr'] 
        df=df.apply(pd.to_numeric)
    return df