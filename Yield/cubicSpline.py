from scipy.interpolate import CubicSpline
import pandas as pd 
import numpy as np

def splineCubic(df) : 
    x = np.array([1/12,2/12,3/12,4/12,6/12,1,2,3,5,7,10,20,30])
    y = df.iloc[0].values
    cs = CubicSpline(x, y)
    x_interp = np.linspace(min(x), max(x), 100)
    y_interp = cs(x_interp)
    return x_interp,y_interp,x,y