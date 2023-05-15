import numpy as np
import random

class LogisticRegression:
    
    def __init__(self):
        self.w_st=None
        self.w=None
        self.score_history=[]
        self.loss_history=[]
        self.score_history_st=[]
        self.loss_history_st=[]
        
    def predict(self, X): #gives us y_hat values for a certain set of weights
        return X@self.w

    def sigmoid(self, x): #expresses the sigmoid function to give the probability of a certain value being true
        return 1/(1+np.exp(-x))
    
    def loss(self, X, y): #stores and returns the average loss per iteration
        p_yhat= self.sigmoid(self.predict(X))
        l=(-y*np.log(p_yhat)- (1-y)*np.log(1-p_yhat)).mean()
        self.loss_history.append(l)
        return l
    
    def score(self, X, y): #stores and returns the score 
        s= 1-self.loss(X,y)
        self.score_history.append(s)
        return s
    
    def fit(self, X, y, alpha, max_epochs=100):
        X=np.concatenate([X, np.ones((X.shape[0], 1))], axis=1) #pads X
        self.w=np.random.rand(len(X[0])) #generates a vector of weights
        for epoch in range (max_epochs):
            y_hat= self.predict(X) #predicted values of y for our current weights
            grad=self.gradient(X, y, y_hat) #calculates the gradient
            self.w-= alpha*grad #changes weights based on the learning rate and the gradient
            loss=self.loss(X, y)
            score=self.score(X, y)

    def gradient(self, X, y, y_pred):
        return (1/len(y))*np.dot(self.sigmoid(y_pred)-y,X)
    
    
    
              
    def predict_st(self, X): #gives us y_hat values for a certain set of weights
        return X@self.w_st
    
    def loss_st(self, X, y): #stores and returns the average loss per iteration
        p_yhat= self.sigmoid(self.predict_st(X))
        l=(-y*np.log(p_yhat)- (1-y)*np.log(1-p_yhat)).mean()
        self.loss_history_st.append(l)
        return l
    
    def score_st(self, X, y): #stores and returns the score 
        s= 1-self.loss_st(X,y)
        self.score_history_st.append(s)
        return s
        
    def fit_stochastic(self, X, y, max_epochs=100, batch_size=10, alpha=0.1):
        X=np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        self.w_st=np.random.rand(len(X[0]))
        n = X.shape[0]
        for epoch in range(max_epochs):
            order = np.arange(n)
            np.random.shuffle(order)
            for batch in np.array_split(order, n // batch_size + 1):
                x_batch = X[batch,:]
                y_batch = y[batch]
                y_batch_pred= self.predict_st(x_batch)
                grad = self.gradient(x_batch, y_batch, y_batch_pred)
                self.w_st -= alpha*grad
                    
            loss=self.loss_st(X, y)
            score=self.score_st(X, y)
        
        
        
        
        
    
    #def p_yhat(self, X):
        #return 1/(1+np.exp(self.predict(X)))

