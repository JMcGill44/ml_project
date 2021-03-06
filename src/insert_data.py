import sqlite3

#connect to the database
conn = sqlite3.connect("./data/database.sqlite")
c = conn.cursor()

# season (2016), day (0), wteam_id, wteam_score (0), lteam_id, lteam_score (0), wlocation, (N), numot (0)
query = "INSERT INTO TourneyCompactResults VALUES (2016, 0, wteam, 0, lteam, 0, 'N', 0);"

#tournament information for the 2016 March Madness Tournament
games = [(1455, 1435),
        (1195, 1192),
        (1276, 1409),
        (1221, 1380),
        (1242, 1122),
        (1163, 1160),
        (1268, 1355),
        (1218, 1143),
        (1455, 1112),
        (1274, 1138),
        (1234, 1396),
        (1437, 1421),
        (1332, 1221),
        (1386, 1153),
        (1463, 1124),
        (1181, 1423),
        (1320, 1400),
        (1401, 1453),
        (1433, 1333),
        (1328, 1167),
        (1314, 1195),
        (1344, 1425),
        (1231, 1151),
        (1246, 1392),
        (1323, 1276),
        (1372, 1452),
        (1458, 1338),
        (1462, 1451),
        (1438, 1214),
        (1139, 1403),
        (1114, 1345),
        (1235, 1233),
        (1211, 1371),
        (1428, 1201),
        (1393, 1173),
        (1292, 1277),
        (1242, 1163),
        (1268, 1218),
        (1274, 1455),
        (1437, 1234),
        (1332, 1386),
        (1181, 1463),
        (1401, 1320),
        (1328, 1433),
        (1314, 1344),
        (1231, 1246),
        (1323, 1372),
        (1458, 1462),
        (1438, 1139),
        (1235, 1114),
        (1211, 1428),
        (1393, 1292),
        (1242, 1268),
        (1437, 1274),
        (1332, 1181),
        (1328, 1401),
        (1314, 1231),
        (1323, 1458),
        (1438, 1235),
        (1393, 1211),
        (1437, 1242),
        (1328, 1332),
        (1314, 1323),
        (1393, 1438),
        (1437, 1328),
        (1314, 1393),
        (1437, 1314)]

#insert the 2016 tournament information into the database
for wteam, lteam in games:
    query = "INSERT INTO TourneyCompactResults VALUES (2016, 0, " + str(wteam) + ", 0, " + str(lteam) +", 0, 'N', 0);"
    c.execute(query)

#commit and close the database connection
conn.commit()
conn.close()
