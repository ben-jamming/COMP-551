import numpy as np

class LinearRegression:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias
        self.N = None
        self.w = None
        pass
    
    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]                         #add a dimension for the features
        self.N = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(self.N)])    #add bias by adding a constant feature of value 1
        #alternatively: self.w = np.linalg.inv(x.T @ x)@x.T@y
        self.w = np.linalg.lstsq(x, y)[0]          #return w for the least square difference
        return self
    
    def predict(self, x):
        if self.add_bias:
            x = np.column_stack([x,np.ones(self.N)])
        yh = x@self.w                            
        return yh
