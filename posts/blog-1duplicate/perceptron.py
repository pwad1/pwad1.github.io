import numpy as np
import random

class Perceptron:
    
    def __init__(self):
        self.w=None
        self.history=[]
    
    def fit (self, X, y, max_steps=1000): 
        X=np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        self.w=np.random.rand(np.size(X[0]))
        y[y==0]=-1
        for t in range (max_steps):
            i=random.randint(0, 99)
            yhat= np.dot(self.w, X[i])
            #print(y[i])
            if yhat*y[i]<0:
                self.w+=y[i]*X[i]
            self.history.append(self.score(X, y))
            if(self.score(X, y)==1):
                break
        
    def predict(self, X):
        #X=np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        return np.where(np.dot(X, self.w)>=0, 1, -1)
    
    def score(self, X, y):
        #X=np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        y_predict=self.predict(X)
        return np.sum(y==y_predict)/len(y)
    
    
        '''
        n_samples, n_features = X.shape
        
        
        self.w=np.random.rand(X.shape[1])
        print(X)
        for t in range (10000):
            
            score=np.dot(X, self.w)*y
            
            print(score)
            misclassified = np.where(score <=0)[0]
            if misclassified.size == 0:
                print("completed")
                break
            
            change = 0.1 * np.dot(X[misclassified].T, y[misclassified])
            
            self.w=self.w+change
            
            accuracy = 1 - (misclassified.size / y.size)
            
            if x%10==0:
                print(x, accuracy, change)
                print(score)
            
            if x%50==0:
                print(self.w)
            '''
    
    
                     
    
        
        '''
        a= np.dot(w,x)+bias
            if a*y<=0:
                w=w+y*x
                bias= bias+y
            if score==1:
                #break
        
        i = np.random.randint(n_samples)
            y_hat_i=np.dot(self.w, X[i])
            
            if y_hat_i*y[i]<0:
                self.w+= y[i]*X[i]  
                
                '''
    '''
    def perceptron_classify(w, b):
        if np.dot(w, x)>b:
            return 0
        else:
            return 1
            
'''