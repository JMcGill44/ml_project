import numpy as np
import metrics

#linear model
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

        #number of columns of X
        x_cols = X.shape[1]

        #initialize weights
        self.weights = np.zeros((x_cols, 1))

        #learning iterations
        for i in range(self.iterations):

            #iterate over data examples (X rows)
            for example_index, example in enumerate(X):

                #example error
                error = example.dot(self.weights) - y[example_index]

                #update weights using error and alpha
                self.weights -= self.alpha * (error * example).reshape(x_cols, 1)


    def test_fit(self, x_train, y_train, x_test, y_test):

        #convert data to numpy arrays of appropriate shape
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train).reshape(x_train.shape[0], 1)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test).reshape(x_test.shape[0], 1)

        #add column of ones for intercept weight
        x_train = np.c_[np.ones(x_train.shape[0]), x_train]
        x_test = np.c_[np.ones(x_test.shape[0]), x_test]

        #number of columns of X
        x_cols = x_train.shape[1]

        #initialize weights
        self.weights = np.zeros((x_cols, 1))

        train_log_loss = []
        test_log_loss = []

        #learning iterations
        for i in range(self.iterations):

            #example errors list
            train_errors = []
            test_errors = []

            #iterate over data examples (X rows)
            for example_index, example in enumerate(x_train):

                #example error
                error = example.dot(self.weights) - y_train[example_index]

                #update weights using error and alpha
                self.weights -= self.alpha * (error * example).reshape(x_cols, 1)

            #calculate the train error
            for train_example_index, train_example in enumerate(x_train):

                train_errors.append(train_example.dot(self.weights) - y_train[train_example_index])

            #calculate the test error
            for test_example_index, test_example in enumerate(x_test):

                test_errors.append(test_example.dot(self.weights) - y_test[test_example_index])
			
			#calculate the logloss
            train_log_loss.append(metrics.log_loss(x_train.dot(self.weights), y_train))
            test_log_loss.append(metrics.log_loss(x_test.dot(self.weights), y_test))

        #return iteration errors
        return (train_log_loss, test_log_loss)


    #make prediction using model weights and input data
    def predict(self, X):

        #convert X to numpy aray
        X = np.asarray(X)

        #add column of ones for intercept value
        X = np.c_[np.ones(X.shape[0]), X]

        #multiply weights by input attributes and sum to get prediction
        prediction = X.dot(self.weights)

        return (prediction)

