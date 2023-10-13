def PayOff(type : str , St : float , K : float) :
    if(type =="Call") : 
        return max(St-K,0)
    elif type=="Put" :
        return(max(K-St,0))