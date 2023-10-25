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
from getImplied import *


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
    sigma = float(data["volatility"])/100
    if len(data["period"])>0 :
        N= int(data["period"])
    else :
        N= 0
    
    
    # Save the figure to a BytesIO object.
    image_stream = io.BytesIO()
    #plt.savefig(image_stream, format='png')
    
     # Move the file pointer to the beginning of the stream.
    

    
    if data["optionType"]=="European" :
        impliedVolatility= np.round(np.array(getReal(ticker,data["maturity"],data["option"],K)).flatten(),3)
        EuroOption = eOp(K=K,ticker=ticker,N=N,ot=data["option"],exp=data["maturity"],d=dividend,r=R,s=sigma)
        EuroOption.D()
        EuroOption.volatility()
        EuroOption.dividend()
        d = EuroOption.d
        St = float(EuroOption.df.iloc[-1,0])
        volatility = round(EuroOption.s*100,3)
        if data["model"]== "Black & Scholes" :
            price = EuroOption.BS()
        elif data["model"]== "Binomial" :
            price = EuroOption.CRR()
            EuroOption.P_CRR(image_stream)
            image_stream.seek(0)
            encoded_image = base64.b64encode(image_stream.read()).decode('utf-8')
        else :
            price = EuroOption.TM()
            EuroOption.P_TM(image_stream)
            image_stream.seek(0)
            encoded_image = base64.b64encode(image_stream.read()).decode('utf-8')
    elif data["optionType"] == "American" :
        impliedVolatility= ["None for American","None for American"]
        UsOption = aOp(K=K,ticker=ticker,N=N,ot=data["option"],exp=data["maturity"],d=dividend, r=R,s=sigma)
        UsOption.D()
        UsOption.volatility()
        UsOption.dividend()
        d = UsOption.d
        St = float(UsOption.df.iloc[-1,0])
        volatility = round(UsOption.s*100,3)
        if data["model"] == "Binomial" : 
            price = UsOption.CRR()
            UsOption.P_CRR(image_stream)
            image_stream.seek(0)
            encoded_image = base64.b64encode(image_stream.read()).decode('utf-8')
        elif data["model"] =="Trinomial" :
            price = UsOption.TM()
            UsOption.P_TM(image_stream)
            image_stream.seek(0)
            encoded_image = base64.b64encode(image_stream.read()).decode('utf-8')
        
    
    return jsonify({'price': round(price,3),"image": encoded_image, "st" :round(St,3), "volatility" :volatility,"impliedVolatility": impliedVolatility[1],"realPrice": impliedVolatility[0], "dividendYield" : round(d,3)  })
    

@app.route('/getDates', methods=['POST'])
def getDates() :
    data= request.get_json()

    expiries = Op.expiries(data["ticker"])

    return jsonify({"expiries" : expiries})




if __name__ == '__main__':
    app.run(debug=True)
