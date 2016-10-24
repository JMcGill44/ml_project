import numpy as np

#linear model #TODO
class Linear_Regression:

    #create model with given parameters
    def __init__(self, alpha, iterations):
        self.weights = []
        self.alpha = alpha
        self.iterations = iterations


    #fit model to given data and return mean squared error for each learning iteration
    def fit(self, X, y):

        #convert data to numpy arrays of appropriate shape
        X = np.asarray(X)
        y = np.asarray(y).reshape(X.shape[0], 1)

        #add column of ones for intercept weight
        X = np.c_[np.ones(X.shape[0]), X]

        #initialize weights
        self.weights = np.zeros((X.shape[1], 1))
        
        #list of iteration errors to return
        mse_values = []
    
        #learning iterations
        for i in range(self.iterations):

            if i % 50 == 0: print(i)

            #example errors list
            errors = []
    
            #iterate over data examples (X rows)
            for example_index, example in enumerate(X):

                #compute example error using current weights
                errors.append(example.dot(self.weights) - y[example_index])

                #iterate over attributes of example
                for attribute_index, attribute in enumerate(example):

                    #update weights
                    weight_adjustment = self.alpha * (errors[example_index] * example[attribute_index])
                    self.weights[attribute_index] = self.weights[attribute_index] - weight_adjustment

            #compute mean squared error for current iteration
            mse_values.append(float(np.square(np.asarray(errors)).sum()) / len(errors))                
    
        #return iteration errors
        return mse_values


    #make prediction using model weights and input data
    def predict(self, X):
        
        #convert X to numpy aray
        X = np.asarray(X)
    
        #add column of ones for intercept value
        X = np.c_[np.ones(X.shape[0]), X]

        #multiply weights by input attributes and sum to get prediction
        return (X.dot(self.weights))        

