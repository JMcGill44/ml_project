import data_module as dm
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import scipy as sp

def logloss(predicted_values, true_values):
    
    logloss = sum(true_values*sp.log(predicted_values) + sp.subtract(1, true_values)*sp.log(sp.subtract(1, predicted_values)))

    logloss = logloss * -1.0 / len(true_values)

    return logloss

#get the train/test data
x_data, y_data = dm.data()

#test data (2003-2014 season)
x_train = []
y_train = []

for x, y in zip(x_data[0:12], y_data[0:12]):
    x_train += x
    y_train += y

#train data (2015 season)
x_test = x_data[12]
y_test = y_data[12]

lm = linear_model.LinearRegression()
lm.fit(x_train, y_train)

y_pred = lm.predict(x_test)
y_t = []
for i in y_pred:
    if i >= 0.5:
        y_t.append(1)
    if i < 0.5:
        y_t.append(0)

print accuracy_score(y_test, y_t)

print logloss(y_pred, y_test)

