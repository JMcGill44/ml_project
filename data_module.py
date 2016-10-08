import sqlite3
import matplotlib.pyplot as plt

#
def season_averages(season, stat):

    teams = 364
    team_id_offset = 1101

    stat_totals   = [0]*teams
    game_totals   = [0]*teams
    stat_averages = [0]*teams

    conn = sqlite3.connect("./data/database.sqlite")
    c = conn.cursor()
     
    query = "SELECT Wteam, sum(w" + stat + "), count(*) FROM RegularSeasonDetailedResults WHERE season = " + str(season) + " GROUP BY wteam"
    c.execute(query)
    results = c.fetchall()

    for wteam_id, stat_sum, games_won in results:
        stat_totals[wteam_id - team_id_offset] += stat_sum
        game_totals[wteam_id - team_id_offset] += games_won

    sql = "SELECT Lteam, sum(l" + stat + "), count(*) FROM RegularSeasonDetailedResults WHERE season = " + str(season) + " GROUP BY lteam"
    c.execute(sql)
    results = c.fetchall()

    for lteam_id, stat_sum, games_lost in results:
        stat_totals[lteam_id - team_id_offset] += stat_sum
        game_totals[lteam_id - team_id_offset] += games_lost

    for i in range(teams):
        if (game_totals[i] != 0):
            stat_averages[i] = stat_totals[i] / game_totals[i]

    return stat_averages


scores           = [x for x in season_averages(2016, "score") if x != 0]
field_goals_made = [x for x in season_averages(2016, "fgm") if x != 0]
turnovers        = [x for x in season_averages(2016, "to") if x != 0]
blocks           = [x for x in season_averages(2016, "blk") if x != 0]


plt.ylim([0, 100])
plt.plot(scores)
plt.plot(field_goals_made)
plt.plot(turnovers)
plt.plot(blocks)
plt.ylabel("Season Average")
plt.xlabel("Team")
plt.show()

