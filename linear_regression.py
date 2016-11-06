import numpy as np
import metrics

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

    
    def test_fit(self, x_train, y_train, x_test, y_test):

        #convert data to numpy arrays of appropriate shape
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train).reshape(x_train.shape[0], 1)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test).reshape(x_test.shape[0], 1)

        #add column of ones for intercept weight
        x_train = np.c_[np.ones(x_train.shape[0]), x_train]
        x_test = np.c_[np.ones(x_test.shape[0]), x_test]

        #initialize weights
        self.weights = np.zeros((x_train.shape[1], 1))
        
        #lists of iteration errors to return
        train_mse_values = []
        test_mse_values = []    

        #train_accuracies = []
        #test_accuracies = []

        train_log_loss = []
        test_log_loss = []

        #learning iterations
        for i in range(self.iterations):

            #if i % 50 == 0: print(i)

            #example errors list
            train_errors = []
            test_errors = []


            #print("\nTest Errors")    
            
            for test_example_index, test_example in enumerate(x_test):

                test_errors.append(test_example.dot(self.weights) - y_test[test_example_index])

                #print(str(test_example.dot(self.weights)) + " - " + str(y_test[test_example_index])
                #      + " = " + str(test_example.dot(self.weights) - y_test[test_example_index]))
    
            #print("\nTrain Errors")

            #iterate over data examples (x_train rows)
            for train_example_index, train_example in enumerate(x_train):

                #compute example error using current weights
                train_errors.append(train_example.dot(self.weights) - y_train[train_example_index])

                #print(str(train_example.dot(self.weights)) + " - " + str(y_train[train_example_index])
                #      + " = " + str(train_example.dot(self.weights) - y_train[train_example_index]))

                #iterate over attributes of example
                for attribute_index, attribute in enumerate(train_example):

                    #update weights
                    weight_adjustment = self.alpha * (train_errors[train_example_index] * train_example[attribute_index])
                    self.weights[attribute_index] = self.weights[attribute_index] - weight_adjustment

            
            #train_accuracies.append(metrics.accuracy(x_train.dot(self.weights), y_train))
            #test_accuracies.append(metrics.accuracy(x_test.dot(self.weights), y_test))

            #print(str(float(np.square(np.asarray(train_errors)).sum())) + " / " + str(len(train_errors))
            #      + " = " + str(float(np.square(np.asarray(train_errors)).sum()) / len(train_errors)))
            
            #print(str(float(np.square(np.asarray(test_errors)).sum())) + " / " + str(len(test_errors))
            #      + " = " + str(float(np.square(np.asarray(test_errors)).sum()) / len(test_errors)))

            #train_log_loss.append(metrics.log_loss(x_train.dot(self.weights), y_train))
            #test_log_loss.append(metrics.log_loss(x_test.dot(self.weights), y_test))

            #compute mean squared train error for current iteration
            #train_mse_values.append(float(np.square(np.asarray(train_errors)).sum()) / len(train_errors))
            #test_mse_values.append(float(np.square(np.asarray(test_errors)).sum()) / len(test_errors))                  

        #return iteration errors
        return (train_log_loss, test_log_loss)


    #make prediction using model weights and input data
    def predict(self, X):
        
        #convert X to numpy aray
        X = np.asarray(X)
    
        #add column of ones for intercept value
        X = np.c_[np.ones(X.shape[0]), X]

        #print(X.shape)
        #print(self.weights.shape)

        #multiply weights by input attributes and sum to get prediction
        return (X.dot(self.weights))        

