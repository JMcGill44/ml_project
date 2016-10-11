import sqlite3

#stats to consider
STATS = ['score', 'ftm', 'or', 'dr', 'ast', 'to', 'stl', 'blk']

#seasons for which data exists
SEASONS = range(2003, 2017)

#offset between 0-index team lists and actual team id
TEAM_ID_OFFSET = 1101

#number of teams
NUM_TEAMS = 364


#extract the per-game stat averages for the relevant stats for each team for a given season
def regular_season_stats(season):

    #list of games played for each team
    game_totals = [0]*NUM_TEAMS

    #list of stat total lists for each team
    stat_totals = []
    for i in range(NUM_TEAMS): stat_totals.append([0]*len(STATS))

    #list of stat average lists for each team
    stat_averages = []
    for i in range(NUM_TEAMS): stat_averages.append([0]*len(STATS))

    #connect to database
    conn = sqlite3.connect("./data/database.sqlite")
    c = conn.cursor()
    
    #create query strings to extract team stat and game totals
    w_query = "SELECT Wteam, "
    l_query = "SELECT Lteam, "

    for stat in STATS:
        w_query += "sum(w" + stat + "), "
        l_query += "sum(l" + stat + "), "

    w_query += "count(*) FROM RegularSeasonDetailedResults WHERE season = " + str(season) + " GROUP BY wteam"
    l_query += "count(*) FROM RegularSeasonDetailedResults WHERE season = " + str(season) + " GROUP BY lteam"
    
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

    #close database connection
    conn.close() 

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

    # return the stat averages for all teams
    return stat_averages


#extract the results of all tournament games for a given season
def tournament_results(season):

    #connect to database
    conn = sqlite3.connect("./data/database.sqlite")
    c = conn.cursor()
     
    #create query string to extract winning and losing team id's for each tournament game for the given season
    query = "SELECT Wteam, Lteam FROM TourneyDetailedResults WHERE season = " + str(season)
    
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
        if results_index % 2 == 0: mixed_results.append((result[1], result[0], 0))

        #odd results - retain order of teams, add '1' label to indicate that the 1st team won
        else: mixed_results.append((result[0], result[1], 1))

    #return the mixed results
    return mixed_results


#TODO
#set up bracket and fill in the initial matchups for the given season's tournmanet
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
    #if team1_id/team2_id is unknown for the given matchup, it
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



    true_bracket = {}

    slots = bracket.keys()
    r1_slots = [slot for slot in slots if len(slot) == 3]
    r2_slots = [slot for slot in slots if 'R1' in slot]
    r3_slots = [slot for slot in slots if 'R2' in slot]
    r4_slots = [slot for slot in slots if 'R3' in slot]
    r5_slots = [slot for slot in slots if 'R4' in slot]
    r6_slots = [slot for slot in slots if 'R5' in slot]
    r7_slots = [slot for slot in slots if 'R6' in slot]

    for slot in r1_slots:

        #create query string to extract winner of the given slot/matchup
        slot_winner_query = ("SELECT Wteam FROM TourneyCompactResults WHERE season = " + str(season) 
                             + " AND ((Wteam = " + str(bracket[slot][0] + TEAM_ID_OFFSET) + " AND Lteam = " 
                             + str(bracket[slot][1] + TEAM_ID_OFFSET) + ") OR (Wteam = " + str(bracket[slot][1] + TEAM_ID_OFFSET) 
                             + " AND Lteam = " + str(bracket[slot][0] + TEAM_ID_OFFSET) + "))")

        #print(slot_winner_query)

        #execute query
        #c.execute(slot_winner_query)
        #results = c.fetchall()

        #print(results)
        
    #close database connection
    conn.close()

    #return dictionary containing bracket information
    return bracket


#use other defined fuctions to get data in the desired form
def data():

    #data lists
    x_data = []
    y_data = []

    #iterate over seasons calling defined functions to extract data
    for season in SEASONS:
    
        #season data lists
        season_x_data = []
        season_y_data = []

        #extract all stats once to avoid redundant queries
        season_stats = regular_season_stats(season)

        #iterate over tournament games, creating data and label lists
        for wteam_id, lteam_id, label in tournament_results(season):

            season_x_data.append(season_stats[wteam_id - TEAM_ID_OFFSET] + season_stats[lteam_id - TEAM_ID_OFFSET])
            season_y_data.append(label)

        #add season data to return list
        x_data.append(season_x_data)
        y_data.append(season_y_data)  

    #return data
    return x_data, y_data


#TODO
def potential_matchups_data(season):
    
    #connect to database
    conn = sqlite3.connect("./data/database.sqlite")
    c = conn.cursor()
     
    #create query string extract seed information for all tournament teams for the given season
    seeds_query = "SELECT Team FROM TourneySeeds WHERE season = " + str(season)
    
    #execute query
    c.execute(seeds_query)
    results = c.fetchall()

    #close database connection
    conn.close()

    #
    matchups_matrix = []
    for i in range(len(results)): matchups_matrix.append([0]*len(results))

    team_indexes = {}

    for index, team_id in enumerate(results):
        
        team_indexes[team_id[0] - TEAM_ID_OFFSET] = index


    


######################################## DEBUGGING OUTPUT #########################################TODO

potential_matchups_data(2015)

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

if True:

    bracket = brackets(2015)

    #for slot in bracket.keys():

    #    print(str(slot) + ": " + str(bracket[slot][0]) + " vs. " + str(bracket[slot][1]))

if False:

    x_data, y_data = data()

    for season_x_data, season_y_data in zip(x_data, y_data):
    
        for game_data, label in zip(season_x_data, season_y_data):
        
            print(str(label) + " - " + str(game_data) + "\n")

