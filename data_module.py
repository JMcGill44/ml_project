import sqlite3
import numpy as np
import linear_regression
import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

#stats to consider
#STATS = ['score', 'ftm', 'or', 'dr', 'ast', 'to', 'stl', 'blk']
#STATS = ['score', 'fgm', 'fga', 'fgm3', 'fga3', 'ftm', 'fta', 'or', 'dr', 'ast', 'to', 'stl', 'blk', 'pf']
STATS = ['fgm', 'to', 'blk', 'or', 'fga', 'stl', 'dr', 'fga3', 'seed']
#TODO

#seasons for which data exists
SEASONS = range(2003, 2017)

#offset between 0-index team lists and actual team id
TEAM_ID_OFFSET = 1101

#number of teams
NUM_TEAMS = 364

#initialize list of team names
TEAMS = ["N/A"]*NUM_TEAMS

#connect to database
conn = sqlite3.connect("./data/database.sqlite")
c = conn.cursor()

#query to extract team information
team_name_query = "SELECT * FROM TEAMS"

#execute team_name_query
c.execute(team_name_query)
results = c.fetchall()

#close database connection
conn.close()

#add team_name info to TEAMS
for team_id, team_name in results: TEAMS[team_id - TEAM_ID_OFFSET] = team_name


#extract the per-game stat averages for the relevant stats for each team for a given season
def regular_season_stats(season):

    #remove seed from the stat list
    season_stats = [stat for stat in STATS if stat != 'seed']

    #number of stats in the stat list (except seed if it's there)
    num_season_stats = len(season_stats)

    #list of number of games played for each team
    game_totals = [0]*NUM_TEAMS

    #list of stat total lists for each team
    stat_totals = []
    for i in range(NUM_TEAMS): stat_totals.append([0]*num_season_stats)

    #list of stat average lists for each team
    stat_averages = []
    for i in range(NUM_TEAMS): stat_averages.append([0]*num_season_stats)

    #list of allowed stat total lists for each team
    allowed_stat_totals = []
    for i in range(NUM_TEAMS): allowed_stat_totals.append([0]*num_season_stats)

    #list of allowed stat average lists for each team
    allowed_stat_averages = []
    for i in range(NUM_TEAMS): allowed_stat_averages.append([0]*num_season_stats)

    #connect to database
    conn = sqlite3.connect("./data/database.sqlite")
    c = conn.cursor()

    #create query strings to extract team stat and game totals
    w_query = "SELECT Wteam, "
    l_query = "SELECT Lteam, "
    w_allowed_query = "SELECT Wteam, "
    l_allowed_query = "SELECT Lteam, "

    for stat in season_stats:
        w_query += "sum(w" + stat + "), "
        l_query += "sum(l" + stat + "), "
        w_allowed_query += "sum(l" + stat + "), "
        l_allowed_query += "sum(w" + stat + "), "

    w_query += "count(*) FROM RegularSeasonDetailedResults WHERE season = " + str(season) + " GROUP BY wteam"
    l_query += "count(*) FROM RegularSeasonDetailedResults WHERE season = " + str(season) + " GROUP BY lteam"
    w_allowed_query += "count(*) FROM RegularSeasonDetailedResults WHERE season = " + str(season) + " GROUP BY wteam"
    l_allowed_query += "count(*) FROM RegularSeasonDetailedResults WHERE season = " + str(season) + " GROUP BY lteam"

    #execute win query - extract information from games won by each time
    c.execute(w_query)
    w_results = c.fetchall()

    #iterate over w_query results - one for each team that won at least one game
    for result in w_results:

        #convert tuple to list
        result_list = list(result)

        #calculate team index from query result and offset
        team_index = result_list[0] - TEAM_ID_OFFSET

        #add number of games won to game totals list
        game_totals[team_index] = result_list[-1]

        #set stat totals to the stats from games won
        stat_totals[team_index] = result_list[1:-1]

    #execute loss query - extract information from games lost by each team
    c.execute(l_query)
    l_results = c.fetchall()

    #iterate over l_query results - one for each team that lost at least one game
    for result in l_results:

        #convert tuple to list
        result_list = list(result)

        #calculate team index from query result and offset
        team_index = result_list[0] - TEAM_ID_OFFSET

        #add number of games lost to game totals list
        game_totals[team_index] += result_list[-1]

        #iterate over result stats, adding them to the appropriate stat totals
        for stat_index, stat in enumerate(result_list[1:-1]):

            stat_totals[team_index][stat_index] += stat

    #calculate stat averages using extracted stat and game totals
    for team_index in range(NUM_TEAMS):

        #if stats exist for this team
        if (game_totals[team_index] != 0):

            #divide all team stats by the number of games that team played to obtain stat averages
            stat_averages[team_index] = [float(stat_total) / game_totals[team_index] for stat_total in stat_totals[team_index]]

    #execute win allowed query - extract opponent information from games won by each time
    c.execute(w_allowed_query)
    w_allowed_results = c.fetchall()

    #iterate over w_query results - one for each team that won at least one game
    for result in w_allowed_results:

        #convert tuple to list
        result_list = list(result)

        #calculate team index from query result and offset
        team_index = result_list[0] - TEAM_ID_OFFSET

        #set allowed stat totals to the stats from games won
        allowed_stat_totals[team_index] = result_list[1:-1]

    #execute loss allowed query - extract information from games lost by each team
    c.execute(l_allowed_query)
    l_allowed_results = c.fetchall()

    #close database connection
    conn.close()

    #iterate over l_allowed_query results - one for each team that lost at least one game
    for result in l_allowed_results:

        #convert tuple to list
        result_list = list(result)

        #calculate team index from query result and offset
        team_index = result_list[0] - TEAM_ID_OFFSET

        #iterate over result stats, adding them to the appropriate stat totals
        for stat_index, stat in enumerate(result_list[1:-1]):

            allowed_stat_totals[team_index][stat_index] += stat

    #calculate stat averages using extracted allowed stat and game totals
    for team_index in range(NUM_TEAMS):

        #if stats exist for this team
        if (game_totals[team_index] != 0):

            #divide all team stats by the number of games that team played to obtain stat averages
            allowed_stat_averages[team_index] = [float(stat_total) / game_totals[team_index] for stat_total in allowed_stat_totals[team_index]]

    #calculate stat differentials (stat - allowed stat)
    stat_differentials = np.asarray(stat_averages) - np.asarray(allowed_stat_averages)

    #normalize the stat averages for each team
    for stat in range(stat_differentials.shape[1]):

        mean = np.mean(stat_differentials[:, stat])
        std = np.std(stat_differentials[:, stat])
        stat_differentials[:, stat] = (stat_differentials[:, stat] - mean) / std

    stat_differentials = stat_differentials.tolist()

    #return the stat averages for all teams
    return stat_differentials


#extract the results of all tournament games for a given season
def tournament_results(season):

    #connect to database
    conn = sqlite3.connect("./data/database.sqlite")
    c = conn.cursor()

    #create query string to extract winning and losing team id's for each tournament game for the given season
    #query = "SELECT Wteam, Lteam FROM TourneyCompactResults WHERE season = " + str(season)

    query = ("""SELECT r.wteam, s1.seed, r.lteam, s2.seed
               FROM TourneyCompactResults r, TourneySeeds s1, TourneySeeds s2
               WHERE r.season = """ + str(season) + " and s1.season = " + str(season) + " AND s2.season = " + str(season) +
               " AND r.wteam = s1.team AND r.lteam = s2.team;")

    #execute query
    c.execute(query)
    results = c.fetchall()

    #close database connection
    conn.close()

    #mix up order of teams so winning team isn't always 1st and label isn't always 1
    mixed_results = []

    #flip order of every other result and add appropriate data label
    for results_index, result in enumerate(results):

        #even results - flip order of teams, add '0' label to indicate that the 1st team lost
        if results_index % 2 == 0: mixed_results.append((result[2], result[3], result[0], result[1], 0))

        #odd results - retain order of teams, add '1' label to indicate that the 1st team won
        else: mixed_results.append((result[0], result[1], result[2], result[3], 1))

    #return the mixed results
    return mixed_results


#set up bracket w/ the initial matchups for the given season's tournament
#return this initial bracket along with actual tournament results
def brackets(season):

    #connect to database
    conn = sqlite3.connect("./data/database.sqlite")
    c = conn.cursor()

    #create query string extract seed information for all tournament teams for the given season
    seeds_query = "SELECT Seed, Team FROM TourneySeeds WHERE season = " + str(season)

    #execute query
    c.execute(seeds_query)
    results = c.fetchall()

    #dictionary containing seed,team_id key,value pairs
    seeded_teams = {}

    #iterate over seeds_query results - build dictionary
    for seed, team_id in results:
        seeded_teams[seed] = team_id - TEAM_ID_OFFSET

    #create query string to extract tournament slots/matchups information
    slots_query = "SELECT Slot, Strongseed, Weakseed FROM TourneySlots WHERE season = " + str(season)

    #execute query
    c.execute(slots_query)
    results = c.fetchall()

    #dictionary with slot,[team1_id, team2_id] key,value pairs
    #if team1_id/team2_id is unknown for the given matchup, it will
    #instead be the slot of the game that will determine the team
    bracket = {}

    #iterate over slots_query results - build dictionary
    for slot, s_seed, w_seed in results:

        #if the game participants are known - replace seed with the appropriate team_id
        if s_seed in seeded_teams:
            s_seed = seeded_teams[s_seed]

        if w_seed in seeded_teams:
            w_seed = seeded_teams[w_seed]

        bracket[slot] = [s_seed, w_seed]

    #initialize dictionary of (team,wins) key,value pairs
    games_won = {}
    for team in seeded_teams.values(): games_won[team] = 0

    games_won_query = ("SELECT Wteam, COUNT(*) FROM TourneyCompactResults WHERE season = "
                       + str(season) + " GROUP BY Wteam")

    #execute games_won query
    c.execute(games_won_query)
    results = c.fetchall()

    #close database connection
    conn.close()

    #set values of games_won w/ query results
    for team_id, wins in results:
        games_won[team_id - TEAM_ID_OFFSET] = wins

    #results of bracket containing (slot,winner) key,value pairs
    bracket_results = {}

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

    #teams that participated in an additional tournament "round"
    play_in_teams = []

    for slot in rounds[0]:
        play_in_teams.append(bracket[slot][0])
        play_in_teams.append(bracket[slot][1])

    #iterate over tournament rounds, updating game winners
    for round_slots in rounds:

        #iterate over round games, determining winner of each
        for slot in round_slots:

            #game participants
            team_1 = bracket[slot][0]
            team_2 = bracket[slot][1]

            #if the game depends on a previous game's results, get those results
            if team_1 not in range(NUM_TEAMS): team_1 = bracket_results[team_1][0]
            if team_2 not in range(NUM_TEAMS): team_2 = bracket_results[team_2][0]

            #win totals for game participants
            team_1_wins = games_won[team_1]
            team_2_wins = games_won[team_2]

            #necessary adjustment for play_in_teams
            if team_1 in play_in_teams: team_1_wins -= 1
            if team_2 in play_in_teams: team_2_wins -= 1

            #determine game winner and update bracket results accordingly
            if (team_1_wins > team_2_wins): bracket_results[slot] = [team_1]
            else: bracket_results[slot] = [team_2]

    #iterate over games
    for slot in slots:

        slot_winner = bracket_results[slot][0]

        #if a play-in team won the game
        if slot_winner in play_in_teams:

            #add other play-in team to game winners list for scoring
            play_in_index = play_in_teams.index(slot_winner)

            if play_in_index % 2 == 0: bracket_results[slot].append(play_in_teams[play_in_index + 1])
            else: bracket_results[slot].append(play_in_teams[play_in_index - 1])

    #return dictionaries containing bracket information
    return bracket, bracket_results


#use other defined fuctions to get data in the desired form
def data(test_season):

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    #iterate over seasons calling defined functions to extract data
    for season in SEASONS:

        #extract all stats once to avoid redundant queries
        season_stats = regular_season_stats(season)

        #iterate over tournament games, creating data and label lists
        for team1_id, team1_seed, team2_id, team2_seed, label in tournament_results(season):

            #convert seed string to appropriate int
            team1_seed = team1_seed[1:]
            team2_seed = team2_seed[1:]
            if len(team1_seed) > 2: team1_seed = 17
            if len(team2_seed) > 2: team2_seed = 17
            team1_seed = int(team1_seed)
            team2_seed = int(team2_seed)

            #build test set
            if season == test_season:

                if 'seed' in STATS:
                    x_test.append(season_stats[team1_id - TEAM_ID_OFFSET] + [team1_seed] + season_stats[team2_id - TEAM_ID_OFFSET] + [team2_seed])
                else:
                    x_test.append(season_stats[team1_id - TEAM_ID_OFFSET] + season_stats[team2_id - TEAM_ID_OFFSET])

                y_test.append(label)

            #build train set
            else:

                if 'seed' in STATS:
                    x_train.append(season_stats[team1_id - TEAM_ID_OFFSET] + [team1_seed] + season_stats[team2_id - TEAM_ID_OFFSET] + [team2_seed])
                else:
                    x_train.append(season_stats[team1_id - TEAM_ID_OFFSET] + season_stats[team2_id - TEAM_ID_OFFSET])

                y_train.append(label)

    #return data
    return x_train, y_train, x_test, y_test



#TODO####################################### DEBUGGING OUTPUT #########################################

if False:

    data(2015)

if False:

    for season in SEASONS:

        print("--------------------------- " + str(season) + " SEASON ---------------------------")

        for team_index, stat_averages in enumerate(regular_season_stats(season)):

            print("\nTeam: " + str(team_index + TEAM_ID_OFFSET))

            for stat_index, stat_average in enumerate(stat_averages):

                print(str(STATS[stat_index]) + "/game : " + str(stat_average))

if False:

    for season in SEASONS[:-1]:

        print("--------------------------- " + str(season) + " SEASON ---------------------------")

        for team1, team2, label in tournament_results(season):

            print(str(team1) + " - " + str(team2) + " : " + str(label))

if False:

    bracket, true_bracket = brackets(2015)

    for slot in bracket.keys():

        print(str(slot) + ": " + str(bracket[slot][0]) + " vs. " + str(bracket[slot][1]))

    print("\n")

    for slot in true_bracket.keys():

        print(str(slot) + ": ", end = "")

        for winner in true_bracket[slot]:

            print(TEAMS[winner] + ", ", end = "")

        print("")

#feature selection
if True:

    #stats list to test
    STATS = []
    remaining_stats = ['score', 'fgm', 'fga', 'fgm3', 'fga3', 'ftm', 'fta', 'or', 'dr', 'ast', 'to', 'stl', 'blk', 'pf', 'seed']

    while len(remaining_stats) > 0:

        print("Remaining Stats: " + str(len(remaining_stats)))

        #running stat totals
        avgs = [0]*len(remaining_stats)

        #loop over the seasons and average the metric scores for each stat
        for test_season in SEASONS[:-1]:

            print("Season: " + str(test_season))

            for index, test_stat in enumerate(remaining_stats):

                #add the stat to the stats list to test
                STATS.append(test_stat)

                #get the data using the current test season and stats list
                x_train, y_train, x_test, y_test = data(test_season)

                #our linear model
                #lm = linear_regression.Linear_Regression(alpha = .00001, iterations = 100)
                #train_errors, test_errors = lm.test_fit(x_train, y_train, x_test, y_test)

                #Sklearn's linear model
                lm = LinearRegression()
                lm.fit(x_train, y_train)

                #predict the test set values
                y_pred = lm.predict(x_test)
                y_pred = y_pred.reshape(y_pred.shape[0], 1)

                #calculate the appropriate metric value and add it to the running total
                avgs[index] += metrics.log_loss(y_pred, np.asarray(y_test).reshape(len(y_test), 1), 0.17276923076923073)
                #avgs[index] += metrics.accuracy(y_pred, np.asarray(y_test).reshape(len(y_test), 1))

                #delete the stat from the stats list in order to test the next one
                del STATS[-1]

        #average the stat sums over the number of seasons
        avgs = [float(stat_sum)/len(SEASONS[:-1]) for stat_sum in avgs]

        #find the best value (min for logloss, max for accuracy)
        best_val = min(avgs)
        #best_val = max(avgs)

        #add the best value to the current stat list
        STATS.append(remaining_stats[avgs.index(best_val)])
        print("\n" + str(remaining_stats[avgs.index(best_val)]) + ": " + str(best_val))
        print(STATS)

        #delete the best value from the remaining stat list
        del remaining_stats[avgs.index(best_val)]


