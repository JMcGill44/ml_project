import sqlite3
import matplotlib.pyplot as plt

# returns an average for the stat parameter 
# over the season parameter
def season_averages(season):

    stats = ['fga3', 'fgm3', 'ast', 'to', 'stl', 'blk']

    teams = 364
    team_id_offset = 1101

    stat_totals   = [0]*teams
    game_totals   = [0]*teams
    stat_averages = [0]*teams

    conn = sqlite3.connect("./data/database.sqlite")
    c = conn.cursor()
     
    query1 = "SELECT Wteam, "
    query2 = "SELECT Lteam, "

    for s in stats:
        query1 += "sum(w" + s + "), "
        query2 += "sum(l" + s + "), "

    query1 += "count(*) FROM RegularSeasonDetailedResults WHERE season = " + str(season) + " GROUP BY wteam"

    query2 += "count(*) FROM RegularSeasonDetailedResults WHERE season = " + str(season) + " GROUP BY lteam"
    
    c.execute(query1)
    results = c.fetchall()
   

    for result in results:

        result_list = list(result)

        team_index = result_list[0] - team_id_offset

        game_totals[team_index] = result_list[-1]
        
        stat_totals[team_index] = result_list[1:-1]


    c.execute(query2)
    results = c.fetchall()

    for result in results:

        result_list = list(result)

        team_index = result_list[0] - team_id_offset

        game_totals[team_index] += result_list[-1]

        for stat_index, stat in enumerate(result_list[1:-1]):
            
            stat_totals[team_index][stat_index] += stat            


    for team_index in range(teams):

        if (game_totals[team_index] != 0):

            stat_dict = {}
    
            for stat_index, stat_total in enumerate(stat_totals[team_index]):

                stat_dict[stats[stat_index]] = float(stat_total) / game_totals[team_index]

            stat_averages[team_index] = stat_dict
  
    for sa in stat_averages:
        print sa


    return stat_averages

for sa in season_averages(2016):
    print sa
#scores           = [x for x in season_averages(2016, "score") if x != 0]
#field_goals_made = [x for x in season_averages(2016, "fgm") if x != 0]
#turnovers        = [x for x in season_averages(2016, "to") if x != 0]
#blocks           = [x for x in season_averages(2016, "blk") if x != 0]


#plt.ylim([0, 100])
#plt.plot(scores)
#plt.plot(field_goals_made)
#plt.plot(turnovers)
#plt.plot(blocks)
#plt.ylabel("Season Average")
#plt.xlabel("Team")
#plt.show()

