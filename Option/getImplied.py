import pandas as pd 
import yoptions as yo 

def getReal(ticker,expiry,ot,K) :
    if ot=="Call" :
        D = pd.DataFrame(yo.get_plain_option(ticker,expiry,"c",K))
    else :
         D = pd.DataFrame(yo.get_plain_option(ticker,expiry,"p",K))
    return [D.iloc[0,3],D.iloc[0,-1]]