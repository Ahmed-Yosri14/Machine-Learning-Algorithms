import numpy as np

class LogisticRegression:
    def __init__(self,lr =0.01,theta=None,max_iters=1000):
        self.theta =theta
        self.max_iters=max_iters
        self.lr = lr


    def fit(self,X,y):
        m,n = X.shape
        X =np.c_[np.ones(m),X]
        if self.theta is None:
            self.theta =np.zeros(n+1)
        for _ in range (self.max_iters):
            z = np.dot(X,self.theta)
            pred = self.sigmoid(z)
            gradient = (1/m)*np.dot(X.T,pred-y)
            self.theta-= self.lr*gradient

        pass

    def predict(self,X):
        m= X.shape[0]
        X =np.c_[np.ones(m),X]
        pred = self.sigmoid(np.dot(X,self.theta))
        return (pred>=0.5).astype(int)
    def sigmoid(self,z):
        return (1/(1+np.exp(-z)))
