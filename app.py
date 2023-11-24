import io
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
from flask_cors import CORS
import base64
from Option.Classes import *
from Option.euoption import *
from Option.amoption import *
from Option.getImplied import *
from Option.plotgreeks import *
from Yield.cubicSpline import *
from Yield.nelsonSiegel import *
from Yield.nelsonSvenson import * 
from Yield.vasicek import *
from Yield.cir import *
from Yield.Scrapper import *
from Yield.Interpolation import * 
from Yield.ModelisationGARCH import *
from Yield.BDscrapper import *
from datetime import datetime
import json

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
    [delta,gamma,vega,rho,theta] = np.zeros(5)

    encoded_image=""
    if data["optionType"]=="European" :
        impliedVolatility= getReal(ticker,data["maturity"],data["option"],K)
        EuroOption = eOp(K=K,ticker=ticker,N=N,ot=data["option"],exp=data["maturity"],d=dividend,r=R,s=sigma)
        EuroOption.D()
        EuroOption.volatility()
        EuroOption.dividend()
        d = EuroOption.d
        St = float(EuroOption.df.iloc[-1,0])
        volatility = round(EuroOption.s*100,3)
        if data["model"]== "Black & Scholes" :
            price = EuroOption.BS()
            delta = EuroOption.delta()
            gamma = EuroOption.gamma()
            vega = EuroOption.vega()
            rho = EuroOption.rho()
            theta = EuroOption.theta()
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
        impliedVolatility= ["0","0"]
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
        
    
    return jsonify({'price': round(price,3),"image": encoded_image, "st" :round(St,3), "volatility" :volatility,"impliedVolatility": impliedVolatility[1],"realPrice": impliedVolatility[0], "dividendYield" : round(d*100,3),"delta" :round(delta,2),"gamma" : round(gamma,2),"vega" :round(vega,2),"rho" :round(rho,2), "theta" : round(theta,2)  })
    

@app.route('/getDates', methods=['POST'])
def getDates() :
    data= request.get_json()

    expiries = Op.expiries(data["ticker"])

    return jsonify({"expiries" : expiries})

@app.route('/getPlot', methods=['POST'])
def getPlot() : 
    data= request.get_json()
    x=[]
    y=[]
    plotOption= P_eOp(K=float(data["strike"]),ticker=data["ticker"],N=0,ot=data["ot"],exp=data["exp"],r=float(data["r"]),d=float(data["d"]),s=float(data["s"]))
    plotOption.D()
    if data["z"]=="None" :
        x,Y=plotOption.P2_OP(float(data["xmin"]),float(data["xmax"]),data["x"])
        match data["y"] :
            case "BS" :
                y=Y[0].tolist()
            case "delta" :
                y=Y[1].tolist()
            case "gamma" :
                y=Y[2].tolist()
            case "rho" :
                y=Y[3].tolist()
            case "theta" :
                y=Y[4].tolist()
            case "vega" :
                y=Y[5].tolist()
        return jsonify({"x" :x.tolist() , "y" : y})
    else : 
        x,z =plotOption.P3_OP(float(data["xmin"]),float(data["xmax"]),float(data["ymin"]),float(data["ymax"]),data["x"],data["y"],data["z"])
        return jsonify({"x" :x[0].tolist() , "y" : x[1].tolist(), "z" : z.tolist()})


@app.route("/getMatrix",methods=["POST"])
def getMatrix() :
    Tickers = request.get_json()
    return jsonify({"response" : "all good !"})


@app.route("/yieldPlot",methods=["POST"])
def yieldPlot() :
    
    data= request.get_json()
    startDate = data["startDate"]
    endDate = data["endDate"]
    country = data["country"]
    x_est,y_est,x,y =  [[],[],[],[]]


    if data["type"] == "Interpolation" : 
        var = Interpolation(country,startDate)

        match data["model"] :
            case "Cubic Spline" :
                var.SC(20)
                x_est = var.sc["maturity"].values
                y_est= var.sc[var.CC+"_TBA"].values
                x = var.data["maturity"].values
                y = var.data[var.CC+"_TBA"].values    

            case  "Nelson-Siegel" :
                var.NS()
                x_est = var.ns["maturity"].values
                y_est= var.ns[var.CC+"_TBA"].values
                x = var.data["maturity"].values
                y = var.data[var.CC+"_TBA"].values

            case  "Nelson-Siegel-Svenson" :
                var.NSS()
                x_est = var.nss["maturity"].values
                y_est= var.nss[var.CC+"_TBA"].values
                x = var.data["maturity"].values
                y = var.data[var.CC+"_TBA"].values
            
            case "Bootsrapping" : 
                var.bootsrapping()
                x_est = var.BS["maturity"].values
                y_est= var.BS[var.CC+"_TBA"].values
                x = var.data["maturity"].values
                y = var.data[var.CC+"_TBA"].values

            case "Cubic-Interpolation" : 
                var.CI()
                x_est = var.ci["maturity"].values
                y_est= var.ci[var.CC+"_TBA"].values
                x = var.data["maturity"].values
                y = var.data[var.CC+"_TBA"].values
            case "Linear-Interpolation" : 
                var.LI()
                x_est = var.li["maturity"].values
                y_est= var.li[var.CC+"_TBA"].values
                x = var.data["maturity"].values
                y = var.data[var.CC+"_TBA"].values
    else :

        var = ModelisationGARCH(country,startDate,endDate)
        a = ModelisationGARCH(country,endDate,datetime.today().strftime("%d-%m-%Y"))
        w = pd.concat([var.data,a.data]).drop_duplicates()
        y = w[var.CC+"_TBA"].values
        x = w.index.astype(str, copy = False)
        var.VGARCH()
        df = var.VASICEKGPMCS(int(data["simulations"]))
        x_est = df.index.astype(str, copy = False)
        
        if data["pathType"] == "Mean" : 
            y_est = np.mean(df.to_numpy().T,axis=0)
        elif data["pathType"] == "Median" :
            y_est = pd.DataFrame(df.to_numpy().T).sort_values(df.columns[-1]).to_numpy()[int(int(data["simulations"])/2)]
        else : 
            y_est = df.to_numpy().T
            
        
    return jsonify({"x_est" : x_est.tolist(),"y_est" : y_est.tolist(), "x" : x.tolist() , "y" : y.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
