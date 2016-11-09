import data_module
import metrics
import linear_regression
import random_forest
import regression_tree
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

#number of brackets to generate for bracket scoring
NUM_BRACKETS = 1000


#create a tournament bracket for the given season using the given model for predictions
def create_and_score_bracket(model, season):

    #get regular season stats, bracket, and bracket results for given season
    season_stats = data_module.regular_season_stats(season)
    bracket, bracket_results = data_module.brackets(season)

    #all slot names
    slots = bracket.keys()

    #create list of slots grouped by tournament round
    rounds = []
    rounds.append([slot for slot in slots if len(slot) == 3])
    rounds.append([slot for slot in slots if 'R1' in slot])
    rounds.append([slot for slot in slots if 'R2' in slot])
    rounds.append([slot for slot in slots if 'R3' in slot])
    rounds.append([slot for slot in slots if 'R4' in slot])
    rounds.append([slot for slot in slots if 'R5' in slot])
    rounds.append([slot for slot in slots if 'R6' in slot])

    #initialize total score and best score/bracket w/ appropriate values
    score_total = 0
    best_score = 0
    best_bracket = {}

    #create 'num_brackets' brackets and keep bracket w/ best score
    for i in range(NUM_BRACKETS):

        #if i % 1000 == 0: print(i)

        #initialize predicted bracket
        predicted_bracket = {}

        #iterate over tournament rounds, updating game winners
        for round_slots in rounds:

            #iterate over round games, determining winner of each
            for slot in round_slots:

                #game participants
                team_1 = bracket[slot][0]
                team_2 = bracket[slot][1]

                #if the game depends on a previous game's results, get those results
                if team_1 not in range(data_module.NUM_TEAMS): team_1 = predicted_bracket[team_1]
                if team_2 not in range(data_module.NUM_TEAMS): team_2 = predicted_bracket[team_2]

                #predict game result
                p = model.predict(np.asarray(season_stats[team_1] + season_stats[team_2]).reshape(1, -1))

                #edge cases for probabilities
                if (p < 0): p = 0
                if (p > 1): p = 1

                #flip weighted coin
                pred = np.random.binomial(1, p)

                #advance predicted winner
                if pred == 1: predicted_bracket[slot] = team_1
                else: predicted_bracket[slot] = team_2

        #score bracket
        score = metrics.bracket_score(predicted_bracket, bracket_results)

        #update total score
        score_total += score

        #update best score/bracket
        if score > best_score:
            best_score = score
            best_bracket = predicted_bracket

    #return best predicted bracket, the corresponding bracket score, and average bracket score
    return best_bracket, best_score, (float(score_total) / NUM_BRACKETS)


######################################TODO######################################
test_season = 2015

#get the train/test data
x_train, y_train, x_test, y_test = data_module.data(test_season)


if False:

    iters = 1000
    alpha = .00001
    accuracy_sum = 0
    log_loss_sum = 0
    best_bracket_sum = 0
    avg_bracket_sum = 0

    for test_season in data_module.SEASONS[:-1]:

        print(test_season)
        x_train, y_train, x_test, y_test = data_module.data(test_season)

        lm = linear_regression.Linear_Regression(alpha = alpha, iterations = iters)
        train_errors, test_errors = lm.test_fit(x_train, y_train, x_test, y_test)
        y_pred = lm.predict(x_test)

        #lm = linear_model.LinearRegression()
        #lm.fit(x_train, y_train)
        #y_pred = np.asarray(lm.predict(x_test)).reshape(len(y_test), 1)

        accuracy_sum += metrics.accuracy(y_pred, np.asarray(y_test).reshape(len(y_test), 1))
        log_loss_sum += metrics.log_loss(y_pred, np.asarray(y_test).reshape(len(y_test), 1))

        #print("\nLinear Model")
        #print("Accuracy: " + str(metrics.accuracy(y_pred, np.asarray(y_test).reshape(len(y_test), 1))))
        #print("LogLoss:  " + str(metrics.log_loss(y_pred, np.asarray(y_test).reshape(len(y_test), 1))))

        best_bracket, best_bracket_score, avg_bracket_score = create_and_score_bracket(lm, test_season)

        best_bracket_sum += best_bracket_score
        avg_bracket_sum += avg_bracket_score

        #print("Best Score: " + str(best_bracket_score))
        #print("Avg Score : " + str(avg_bracket_score))

        #for game in sorted(best_bracket):

        #    print(str(game) + " : " + str(data_module.TEAMS[best_bracket[game]]))

        #plt.plot(train_errors)
        #plt.plot(test_errors)
        #plt.ylim([0.24, 0.3])
        #plt.show()
    print("Average Accuracy: " + str(float(accuracy_sum)/len(data_module.SEASONS[:-1])))
    print("Average Log Loss: " + str(float(log_loss_sum)/len(data_module.SEASONS[:-1])))
    print("Average Best Bracket: " + str(float(best_bracket_sum)/len(data_module.SEASONS[:-1])))
    print("Average Avg Bracket: " + str(float(avg_bracket_sum)/len(data_module.SEASONS[:-1])))

if True:

    accuracy_sum = 0
    log_loss_sum = 0
    best_bracket_sum = 0
    avg_bracket_sum = 0
    for test_season in data_module.SEASONS[:-1]:

        print("\n" + str(test_season))

        x_train, y_train, x_test, y_test = data_module.data(test_season)
        rf = regression_tree.RegressionTree(3, 0)
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test)

        accuracy_sum += metrics.accuracy(y_pred, np.asarray(y_test).reshape(len(y_test), 1))
        log_loss_sum += metrics.log_loss(y_pred, np.asarray(y_test).reshape(len(y_test), 1))


        #rf.print_tree()

        #print("\nDecision Tree")
        print("Accuracy: " + str(metrics.accuracy(y_pred, np.asarray(y_test).reshape(len(y_test), 1))))
        print("LogLoss:  " + str(metrics.log_loss(y_pred, np.asarray(y_test).reshape(len(y_test), 1))))

        best_bracket, best_bracket_score, avg_bracket_score = create_and_score_bracket(rf, test_season)
        best_bracket_sum += best_bracket_score
        avg_bracket_sum += avg_bracket_score
        print("Best Score: " + str(best_bracket_score))
        print("Avg Score : " + str(avg_bracket_score))

    print("Average Accuracy: " + str(float(accuracy_sum)/len(data_module.SEASONS[:-1])))
    print("Average Log Loss: " + str(float(log_loss_sum)/len(data_module.SEASONS[:-1])))
    print("Average Best Bracket: " + str(float(best_bracket_sum)/len(data_module.SEASONS[:-1])))
    print("Average Avg Bracket: " + str(float(avg_bracket_sum)/len(data_module.SEASONS[:-1])))

    #for game in sorted(best_bracket):

    #    print(str(game) + " : " + str(data_module.TEAMS[best_bracket[game]]))

if False:
    accuracy = 0
    log_loss = 0
    for test_season in data_module.SEASONS[:-1]:
        print(test_season)
        x_train, y_train, x_test, y_test = data_module.data(test_season)

        rf = random_forest.RandomForest(75, 3, 5)
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test)

        accuracy += metrics.accuracy(y_pred, np.asarray(y_test).reshape(len(y_test), 1))
        log_loss += metrics.log_loss(y_pred, np.asarray(y_test).reshape(len(y_test), 1))

        #print("\nRandom Forest")
        #print("Accuracy: " + str(metrics.accuracy(y_pred, np.asarray(y_test).reshape(len(y_test), 1))))
        #print("LogLoss:  " + str(metrics.log_loss(y_pred, np.asarray(y_test).reshape(len(y_test), 1))))

    print("\nRandom Forest")
    print("Accuracy: " + str(float(accuracy)/len(data_module.SEASONS[:-1])))
    print("LogLoss:  " + str(float(log_loss)/len(data_module.SEASONS[:-1])))

