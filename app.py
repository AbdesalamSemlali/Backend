from flask import Flask, jsonify, request
from flask_cors import CORS
from strToDate import get_date_difference
from blackScholes import BlackScholes
from binomial import BinomialModel
from trinomial import TrinomialModel
from americainBinomial import *
app = Flask(__name__)
CORS(app)


@app.route('/calculate', methods=['POST'])
def calculate():
    #getting the data
    data= request.get_json() 

    #Getting the variables 

    maturity = get_date_difference(data["maturity"])/365
    Sigma = float(data["volatility"])/100
    St=float(data["underlying"])
    K=float(data["strike"])
    R=float(data["interest"])/100

    if len(data["period"])>0 :
        N= int(data["period"])
    price =0
    if data["optionType"]=="Euro" :
        if data["model"]== "Black&Scholes" :
            price = BlackScholes(St=St,K=K, Sigma=Sigma, R=R,T=maturity,type=data["option"])
        elif data["model"]== "Binomial" :
            price = BinomialModel(St=St,K=K, Sigma=Sigma, R=R,T=maturity,type=data["option"],N=N)
        else :
            price = TrinomialModel(St=St,K=K, Sigma=Sigma, R=R,T=maturity,type=data["option"],N=N)
    else :
        price= american_fast_tree(St=St,K=K, Sigma=Sigma, R=R,T=maturity,type=data["option"],N=N)
    return jsonify({'price': round(price,3) })

if __name__ == '__main__':
    app.run(debug=True)
