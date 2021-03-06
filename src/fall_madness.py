import data_module
import metrics
import linear_regression
import random_forest
import regression_tree
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

#number of brackets to generate for bracket scoring
NUM_BRACKETS = 1000

TEAM_ID_OFFSET = 1101

#create a tournament bracket for the given season using the given model for predictions
def create_and_score_bracket(model, season, all_stats):

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

#function to bound the predictions
def bound_predictions(predicted_values, bound_offset):

    #bound predictions between (bound_offset, 1 - bound_offset)
    lower_bound = bound_offset
    upper_bound = 1 - bound_offset

    for predicted_value_index in range(len(predicted_values)):

        if predicted_values[predicted_value_index] <= lower_bound: predicted_values[predicted_value_index] = lower_bound
        if predicted_values[predicted_value_index] >= upper_bound: predicted_values[predicted_value_index] = upper_bound

    return predicted_values


#---------------------- EXPERIMENTS ----------------------#
#Change "if False:" to "if True:" to run the experiments

#Experiment 1 - Cross-Validated Linear Regression
if True:

    #learning parameters
    all_stats = False
    iterations = 1000
    alpha = .00001
    bound_offset = 0.05

    #cross-validation sum variables
    accuracy_sum = 0
    log_loss_sum = 0
    best_bracket_sum = 0
    avg_bracket_sum = 0

    #perform cross-validation on all seasons
    for test_season in data_module.SEASONS:

        print("LR - Test Season = " + str(test_season))

		#create the test/train splits using the current test season as the test set
        x_train, y_train, x_test, y_test = data_module.data(test_season, all_stats)

		#create the linear model, trian on the training data, and predict the test set
        lm = linear_regression.Linear_Regression(alpha, iterations)
        lm.fit(x_train, y_train)
        y_pred = bound_predictions(lm.predict(x_test), bound_offset)

		#calculate the accuracy and log loss for the current test season
        test_season_accuracy = metrics.accuracy(y_pred, np.asarray(y_test).reshape(len(y_test), 1))
        test_season_logloss = metrics.log_loss(y_pred, np.asarray(y_test).reshape(len(y_test), 1))
        accuracy_sum += test_season_accuracy
        log_loss_sum += test_season_logloss

		#generate brackets using the predictions for the current season
        best_bracket, best_bracket_score, avg_bracket_score = create_and_score_bracket(lm, test_season, all_stats)
        best_bracket_sum += best_bracket_score
        avg_bracket_sum += avg_bracket_score

	#print the average metrics for this experiment
    print("\nAverage Linear Regression Results")
    print("Average Accuracy: " + str(float(accuracy_sum)/len(data_module.SEASONS)))
    print("Average Log Loss: " + str(float(log_loss_sum)/len(data_module.SEASONS)))
    print("Average Best Bracket: " + str(float(best_bracket_sum)/len(data_module.SEASONS)))
    print("Average Avg Bracket: " + str(float(avg_bracket_sum)/len(data_module.SEASONS)))

	
#Experiment 2: Cross-Validated Random Forest
if False:

    #learning parameters
    all_stats = True
    num_trees = 75
    depth = 100
    min_samples = 10
    features = 10 
    bound_offset = 0.05

    #cross validation sum variables
    accuracy_sum = 0
    log_loss_sum = 0
    best_bracket_sum = 0
    avg_bracket_sum = 0

    #perform cross-validation on all seasons
    for test_season in data_module.SEASONS:

        print("RF - Test Season = " + str(test_season))

		#create the test/train splits using the current test season as the test set
        x_train, y_train, x_test, y_test = data_module.data(test_season, all_stats)

		#create the random forest, trian on the training data, and predict the test set
        rf = random_forest.RandomForest(num_trees, depth, min_samples, features)
        rf.fit(x_train, y_train)
        y_pred = bound_predictions(rf.predict(x_test), bound_offset)

		#calculate the accuracy and log loss for the current test season
        test_season_accuracy = metrics.accuracy(y_pred, np.asarray(y_test).reshape(len(y_test), 1))
        test_season_logloss = metrics.log_loss(y_pred, np.asarray(y_test).reshape(len(y_test), 1))
        accuracy_sum += test_season_accuracy
        log_loss_sum += test_season_logloss

		#generate brackets using the predictions for the current season
        best_bracket, best_bracket_score, avg_bracket_score = create_and_score_bracket(lm, test_season, all_stats)
        best_bracket_sum += best_bracket_score
        avg_bracket_sum += avg_bracket_score

	#print the average metrics for this experiment
    print("\nAverage Random Forest")
    print("Average Accuracy: " + str(float(accuracy_sum)/len(data_module.SEASONS)))
    print("Average Log Loss: " + str(float(log_loss_sum)/len(data_module.SEASONS)))
    print("Average Best Bracket: " + str(float(best_bracket_sum)/len(data_module.SEASONS)))
    print("Average Avg Bracket: " + str(float(avg_bracket_sum)/len(data_module.SEASONS)))

#Experiment 3: Normalized Data
#Experiment 3 is experiments 1 & 2 with/without the seed feature.
#To run Experiment 4, change the "USE_SEED = True" line at the top of data_module.py to False
#and rerun experiments 1 & 2

#Experiment 4: Normalized Data
#Experiment 4 is experiments 1 & 2 with/without normalization.
#To run Experiment 4, change the "NORMALIZATION = True" line at the top of data_module.py to False
#and rerun experiments 1 & 2