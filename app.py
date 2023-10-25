import io
from flask import Flask, send_file, request, jsonify
import matplotlib.pyplot as plt
from flask_cors import CORS
import base64
from strToDate import get_date_difference
from blackScholes import BlackScholes
from binomial import BinomialModel
from trinomial import TrinomialModel
from americainBinomial import *
from Classes import *
from euoption import *
from amoption import *


app = Flask(__name__)
CORS(app)


@app.route('/calculate', methods=['POST'])
def calculate():

    #getting the data
    data= request.get_json() 
    K=float(data["strike"])
    R=float(data["interest"])/100
    ticker = data["ticker"]
    dividend = float(data["dividend"])/100
    if len(data["period"])>0 :
        N= int(data["period"])
    
    
    # Save the figure to a BytesIO object.
    #image_stream = io.BytesIO()
    #plt.savefig(image_stream, format='png')

     # Move the file pointer to the beginning of the stream.
    #image_stream.seek(0)

    #encoded_image = base64.b64encode(image_stream.read()).decode('utf-8')
    if data["optionType"]=="European" :
        EuroOption = eOp(K=K,ticker=ticker,N=N,ot=data["option"],exp=data["maturity"],d=dividend,r=R)
        EuroOption.D()
        EuroOption.volatility() 
        St = float(EuroOption.df.iloc[-1,0])
        volatility = round(EuroOption.s*100,3)
        if data["model"]== "Black & Scholes" :
            price = EuroOption.BS()
        elif data["model"]== "Binomial" :
            price = EuroOption.CRR()
        else :
            price = EuroOption.TM()
    elif data["optionType"] == "American" :
        UsOption = aOp(K=K,ticker=ticker,N=N,ot=data["option"],exp=data["maturity"],d=dividend, r=R)
        UsOption.D()
        UsOption.volatility()
        St = float(UsOption.df.iloc[-1,0])
        volatility = round(UsOption.s*100,3)
        if data["model"] == "Binomial" : 
            price = UsOption.CRR()
        elif data["model"] =="Trinomial" :
            price = UsOption.TM()
        
    
    return jsonify({'price': round(price,3), "st" :round(St,3), "volatility" :volatility })
    

@app.route('/getDates', methods=['POST'])
def getDates() :
    data= request.get_json()

    expiries = Op.expiries(data["ticker"])

    return jsonify({"expiries" : expiries})




if __name__ == '__main__':
    app.run(debug=True)
