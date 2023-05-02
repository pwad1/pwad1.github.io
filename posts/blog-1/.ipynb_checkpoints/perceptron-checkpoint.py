import numpy as np
import random

class Perceptron:
    
    def __init__(self):
        self.w=None
        self.history=[]
    
    def fit (self, X, y, max_steps=1000): 
        """
        Generates w, an instance of weights. This is initially generated randomly. We then randomly choose a point in X,
        and check whether our perceptron classifies it correctly. If the perceptron classifies it correctly, we do not
        update w, and check whether the accuracy of the perceptron is 100%. If the accuracy is 100%- our job is done.
        If the accuracy is not 100%- we select another random point in X, and check if it is classified correctly.
        If it is classified incorrectly, we update the value of w by the function self.w+=y[i]*X[i]. We repeat this until
        the perceptron gives us a 100% accuracy, or we reach the max steps.
        """
        X=np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        self.w=np.random.rand(np.size(X[0]))
        y[y==0]=-1
        for t in range (max_steps):
            i=random.randint(0, 99)
            yhat= np.dot(self.w, X[i])
            if yhat*y[i]<0:
                self.w+=y[i]*X[i]
            self.history.append(self.score(X, y))
            if(self.score(X, y)==1):
                break
        
    def predict(self, X):
        """
        Returns the predicted labels for all points based on the current 
        """
        return np.where(np.dot(X, self.w)>=0, 1, -1)
    
    def score(self, X, y):
        """
        Returns the proportion of y values classified correctly by the perceptron
        """
        y_predict=self.predict(X)
        return np.sum(y==y_predict)/len(y)