import data_module
import metrics
import linear_regression
import random_forest
import regression_tree
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

from sklearn import linear_model

#number of brackets to generate for bracket scoring
NUM_BRACKETS = 1000

TEAM_ID_OFFSET = 1101

#create a tournament bracket for the given season using the given model for predictions
def create_and_score_bracket(model, season, all_stats):

    ### TODO ###
    #connect to database
    conn = sqlite3.connect("./data/database.sqlite")
    c = conn.cursor()

    #create query string extract seed information for all tournament teams for the given season
    seeds_query = "SELECT Team, Seed FROM TourneySeeds WHERE season = " + str(season)

    #execute query
    c.execute(seeds_query)
    results = c.fetchall()

    #dictionary containing seed,team_id key,value pairs
    team_seeds = {}

    #iterate over seeds_query results - build dictionary
    for team_id, seed in results:

        #convert seed string to appropriate int
        seed = seed[1:]
        if len(seed) > 2: seed = 17
        seed = int(seed)
        team_seeds[team_id - TEAM_ID_OFFSET] = seed

    ### TODO ###

    #get regular season stats, bracket, and bracket results for given season
    season_stats = data_module.regular_season_stats(season, all_stats)
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
                if 'seed' in data_module.STATS:
                    p = model.predict(np.asarray(season_stats[team_1] + [team_seeds[team_1]] + season_stats[team_2] + [team_seeds[team_2]]).reshape(1, -1))
                else:
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
test_season = 2016

#get the train/test data
x_train, y_train, x_test, y_test = data_module.data(test_season, False)

#single Linear Regression
if False:

    #parameters
    test_season = 2016
    all_stats = False
    iters = 1000
    alpha = .00001
    offset = 0.045307692307692306

    x_train, y_train, x_test, y_test = data_module.data(test_season, all_stats)

    lm = linear_regression.Linear_Regression(alpha = alpha, iterations = iters)
    train_errors, test_errors = lm.test_fit(x_train, y_train, x_test, y_test)
    y_pred = lm.predict(x_test)

    best_bracket, best_bracket_score, avg_bracket_score = create_and_score_bracket(lm, test_season, all_stats)

    print("Linear Regression " + str(test_season))
    print("Accuracy: " + str(metrics.accuracy(y_pred, np.asarray(y_test).reshape(len(y_test), 1))))
    print("LogLoss:  " + str(metrics.log_loss(y_pred, np.asarray(y_test).reshape(len(y_test), 1), offset)))
    print("Best Score: " + str(best_bracket_score))
    print("Avg Score : " + str(avg_bracket_score))


#single Random Forest
if False:

    #parameters
    test_season = 2016
    all_stats = True
    num_trees = 500
    depth = 100
    min_samples = 10
    features = 10
    offset = 0.045307692307692306

    x_train, y_train, x_test, y_test = data_module.data(test_season, all_stats)

    rf = random_forest.RandomForest(num_trees, depth, min_samples, features)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)

    best_bracket, best_bracket_score, avg_bracket_score = create_and_score_bracket(rf, test_season, all_stats)

    print("Random Forest " + str(test_season))
    print("Accuracy: " + str(metrics.accuracy(y_pred, np.asarray(y_test).reshape(len(y_test), 1))))
    print("LogLoss:  " + str(metrics.log_loss(y_pred, np.asarray(y_test).reshape(len(y_test), 1), offset)))
    print("Best Score: " + str(best_bracket_score))
    print("Avg Score : " + str(avg_bracket_score))

#cross-validated Linear Regression
if False:

    #parameters
    test_season = 2016
    all_stats = False
    iters = 1000
    alpha = .00001
    offset = 0.045307692307692306

    accuracy_sum = 0
    log_loss_sum = 0
    best_bracket_sum = 0
    avg_bracket_sum = 0

    #perform cross-validation on all seasons
    for test_season in data_module.SEASONS:

        x_train, y_train, x_test, y_test = data_module.data(test_season, all_stats)

        rf = random_forest.RandomForest(num_trees, depth, min_samples, features)
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test)

        test_season_accuracy = metrics.accuracy(y_pred, np.asarray(y_test).reshape(len(y_test), 1)
        test_season_logloss = metrics.log_loss(y_pred, np.asarray(y_test).reshape(len(y_test), 1), offset)
        accuracy_sum += test_season_accuracy
        log_loss_sum += test_season_logloss

        best_bracket, best_bracket_score, avg_bracket_score = create_and_score_bracket(lm, test_season, all_stats)
        best_bracket_sum += best_bracket_score
        avg_bracket_sum += avg_bracket_score

        #print metrics for each test season
        #print("Linear Regression " + str(test_season))
        #print("Accuracy: " + str(test_season_accuracy))
        #print("LogLoss:  " + str(test_season_logloss))
        #print("Best Score: " + str(best_bracket_score))
        #print("Avg Score : " + str(avg_bracket_score) + "\n")

    print("\nAverage Linear Regression")
    print("Average Accuracy: " + str(float(accuracy_sum)/len(data_module.SEASONS)))
    print("Average Log Loss: " + str(float(log_loss_sum)/len(data_module.SEASONS)))
    print("Average Best Bracket: " + str(float(best_bracket_sum)/len(data_module.SEASONS)))
    print("Average Avg Bracket: " + str(float(avg_bracket_sum)/len(data_module.SEASONS)))


#cross-validated Random Forest
if False:

    #parameters
    test_season = 2016
    all_stats = True
    num_trees = 500
    depth = 100
    min_samples = 10
    features = 10
    offset = 0.045307692307692306

    accuracy_sum = 0
    log_loss_sum = 0
    best_bracket_sum = 0
    avg_bracket_sum = 0

    #perform cross-validation on all seasons
    for test_season in data_module.SEASONS:

        x_train, y_train, x_test, y_test = data_module.data(test_season, all_stats)

        rf = linear_regression.Linear_Regression(alpha = alpha, iterations = iters)
        train_errors, test_errors = lm.test_fit(x_train, y_train, x_test, y_test)
        y_pred = lm.predict(x_test)

        test_season_accuracy = metrics.accuracy(y_pred, np.asarray(y_test).reshape(len(y_test), 1)
        test_season_logloss = metrics.log_loss(y_pred, np.asarray(y_test).reshape(len(y_test), 1), offset)
        accuracy_sum += test_season_accuracy
        log_loss_sum += test_season_logloss

        best_bracket, best_bracket_score, avg_bracket_score = create_and_score_bracket(lm, test_season, all_stats)
        best_bracket_sum += best_bracket_score
        avg_bracket_sum += avg_bracket_score

        #print metrics for each test season
        #print("Random Forest " + str(test_season))
        #print("Accuracy: " + str(test_season_accuracy))
        #print("LogLoss:  " + str(test_season_logloss))
        #print("Best Score: " + str(best_bracket_score))
        #print("Avg Score : " + str(avg_bracket_score) + "\n")

    print("\nAverage Random Forest")
    print("Average Accuracy: " + str(float(accuracy_sum)/len(data_module.SEASONS)))
    print("Average Log Loss: " + str(float(log_loss_sum)/len(data_module.SEASONS)))
    print("Average Best Bracket: " + str(float(best_bracket_sum)/len(data_module.SEASONS)))
    print("Average Avg Bracket: " + str(float(avg_bracket_sum)/len(data_module.SEASONS)))

######################################TODO######################################

#offset stuff
if False:

    iters = 1000
    alpha = .00001

    avg_min_offset = 0

    for test_season in data_module.SEASONS:

        x_train, y_train, x_test, y_test = data_module.data(test_season, False)

        rf = linear_regression.Linear_Regression(alpha = alpha, iterations = iters)
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test)

        #accuracy = metrics.accuracy(y_pred, np.asarray(y_test).reshape(len(y_test), 1))
        #log_loss = metrics.log_loss(y_pred, np.asarray(y_test).reshape(len(y_test), 1))

        #print("\nRandom Forest")
        #print("Accuracy: " + str(metrics.accuracy(y_pred, np.asarray(y_test).reshape(len(y_test), 1))))
        min_log_loss = np.inf
        min_offset = np.inf
        for i in range(1, 401):
            offset = 0 + i * .001
            log_loss = metrics.log_loss(y_pred, np.asarray(y_test).reshape(len(y_test), 1), offset)
            #print("LogLoss:  " + str(log_loss))

            if log_loss < min_log_loss:
                min_offset = offset
                min_log_loss = log_loss

        print("\nMin log loss: " + str(min_log_loss))
        print("Min offset: " + str(min_offset))

        avg_min_offset += min_offset
    print("\nAverage Min offset: " + str(float(avg_min_offset)/len(data_module.SEASONS)))

