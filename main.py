import data_module as dm
import numpy as np

from sklearn import linear_model
import matplotlib.pyplot as plt

#calculate the "accuracy" metric of some predictions given lists of the predictions and the true values
def accuracy(predicted_values, true_values):

    #TODO rounding
    #create numpy arrays of true values and rounded predicted values (>= .5 --> 1, 0 otherwise)
    predicted_values = np.array(predicted_values).round()
    true_values = np.array(true_values)
    
    #calculate and return accuracy = number of correctly predicted values / number of values
    return (predicted_values == true_values).sum() / len(predicted_values)


#calculate the "LogLoss" metric of some predictions given lists of the predictions and the true values
def log_loss(predicted_values, true_values):
    
    
    offset = 0.00001
    for i, pred in enumerate(predicted_values):
        if pred > 1:
            predicted_values[i] = 1 - offset
        elif pred < 0:
            predicted_values[i] = 0 + offset 

    
    #calculate numerator of LogLoss
    ll_n = sum(true_values * np.log(predicted_values) + 
               np.subtract(1, true_values) * np.log(np.subtract(1, predicted_values)))
    
    #calculate and return LogLoss
    return (-ll_n) / len(true_values)


#TODO
def bracket_score(predicted_bracket, true_bracket):
    
    score = 0

    #round 1 - 0 points, but either counts..
    
    #TODO
    for game_index, (predicted_winner, true_winners) in enumerate(zip(predicted_bracket, true_bracket)):
        
        if predicted_winner in true_winners:

            if (4 <= game <= 35): score += 10           #10 - round 2
            elif (36 <= game_index <= 51): score += 20  #20 - round 3
            elif (52 <= game_index <= 59): score += 40  #40 - round 4
            elif (60 <= game_index <= 63): score += 80  #80 - round 5
            elif (64 <= game_index <= 65): score += 160 #160 - round 6
            elif game_index == 66: score += 320         #320 - championship

    return score


#TODO
def create_bracket(predictions, season):

    bracket = db.bracket(season)

    return bracket

def normalize(data):

    normalized_data = []

    #TODO skips the 2016 season with data[:-1]
    for season in data[:-1]:

        stats = [[] for i in range(len(season[0]))]
    
        #create a list of each of the stats to find the mean and std
        for index, team in enumerate(season):
            for i in range(len(season[index])):
                stats[i].append(season[index][i])

        #mean and standard deviation for the season
        means = [np.mean(stat) for stat in stats]
        stds = [np.std(stat) for stat in stats]
        
        #list of lists to hold the normalized season data
        normalized_season = [[[] for j in season[0]] for i in season]
    
        #normalize the statistics
        for index, team in enumerate(season):
            for i in range(len(season[index])):
                normalized_season[index][i] = float((season[index][i] - means[i]))/stds[i]

        #add the normalized season to the dataset
        normalized_data.append(normalized_season)
    
    #TODO append an empty list for the 2016 season
    normalized_data.append([])   
    
    #return the normalized data
    return normalized_data

    
#get the train/test data
x_data, y_data = dm.data()

#test data (2003-2014 seasons)
x_train = []
y_train = []

for season_x, season_y in zip(x_data[:12], y_data[:12]):
    x_train += season_x
    y_train += season_y

#test data (2015 season)
x_test = x_data[12]
y_test = y_data[12]



#TODO
lm = linear_model.LinearRegression()
lm.fit(x_train, y_train)
y_pred = lm.predict(x_test)
print("Accuracy: " + str(accuracy(y_pred, y_test)))
print("LogLoss:  " + str(log_loss(y_pred, y_test)))

# plot the accuracy values for each season
acc = []
l = []
for i in range(len(x_data) - 1):
    x = x_data[:i] + x_data[(i + 1):]
    y = y_data[:i] + y_data[(i + 1):]

    x_train = []
    y_train = []

    for season_x, season_y in zip(x, y):
        x_train += season_x
        y_train += season_y
    
    #test data (2015 season)
    x_test = x_data[i]
    y_test = y_data[i]
    
    #TODO
    lm = linear_model.LinearRegression()
    lm.fit(x_train, y_train)
    y_pred = lm.predict(x_test)
    
    acc.append(accuracy(y_pred, y_test))
    l.append(log_loss(y_pred, y_test))


x_data = normalize(x_data)

x_train = []
y_train = []

for season_x, season_y in zip(x_data[:12], y_data[:12]):
    x_train += season_x
    y_train += season_y

#test data (2015 season)
x_test = x_data[12]
y_test = y_data[12]



#TODO
lm = linear_model.LinearRegression()
lm.fit(x_train, y_train)
y_pred = lm.predict(x_test)
print("Accuracy: " + str(accuracy(y_pred, y_test)))
print("LogLoss:  " + str(log_loss(y_pred, y_test)))

# plot the accuracy values for each season
norm_acc = []
norm_l = []
for i in range(len(x_data) - 1):
    x = x_data[:i] + x_data[(i + 1):]
    y = y_data[:i] + y_data[(i + 1):]

    x_train = []
    y_train = []

    for season_x, season_y in zip(x, y):
        x_train += season_x
        y_train += season_y
    
    #test data (2015 season)
    x_test = x_data[i]
    y_test = y_data[i]
    
    #TODO
    lm = linear_model.LinearRegression()
    lm.fit(x_train, y_train)
    y_pred = lm.predict(x_test)
    
    norm_acc.append(accuracy(y_pred, y_test))
    norm_l.append(log_loss(y_pred, y_test))


plt.plot(range(2003, 2016), acc, 'b', lw=2, label="Regular data")
plt.plot(range(2003, 2016), norm_acc, 'r', lw=2, label="Normalized data")
plt.ylim([0, 1])
plt.ylabel("Accuracy")
plt.xlabel("Season")
plt.title("Accuracy for each season")
plt.legend(loc="lower right")
plt.show()

plt.plot(range(2003, 2016), l, 'b', lw=2, label="Regular data")
plt.plot(range(2003, 2016), norm_l, 'r', lw=2, label="Normalized data")
plt.ylim([0, 1])
plt.ylabel("LogLoss")
plt.xlabel("Season")
plt.title("LogLoss for each season")
plt.legend(loc="lower right")
plt.show()
