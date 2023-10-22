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


    fig, ax = plt.subplots()
    ax.plot(np.linspace(-1, 1, 100),np.exp(-R*np.linspace(-1, 1, 100)))  # Adjust this based on your data.

    # Save the figure to a BytesIO object.
    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='png')

     # Move the file pointer to the beginning of the stream.
    image_stream.seek(0)

    encoded_image = base64.b64encode(image_stream.read()).decode('utf-8')


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
    return jsonify({'price': round(price,3), 'image': encoded_image })

if __name__ == '__main__':
    app.run(debug=True)
