import numpy as np
import csv
# parameters set up
Back_82 = 82
Back_41 = 41
Back_10 = 10
Back_3 = 3
HOME_AWAY_BACK_2 = 2

# columns names set up
Original_COLUMNS = ['GAME_DATE_EST','GAME_ID','HOME_TEAM_ID','VISITOR_TEAM_ID','pointspread','PTS_home',
                    'FGM_home','FGA_home','FG_PCT_home','FG2M_home','FG2A_home','FG2_PCT_home','FTM_home',
                    'FTA_home','FT_PCT_home','FG3M_home','FG3A_home','FG3_PCT_home','AST_home','OREB_home',
                    'DREB_home','REB_home','STL_home','BLK_home','TO_home','PF_home','PLUS_MINUS_home',
                    'STL/TO_home','EFG%_home','PPS_home','FIC_home','PTS_away','FGM_away','FGA_away',
                    'FG_PCT_away','FG2M_away','FG2A_away','FG2_PCT_away','FTM_away','FTA_away','FT_PCT_away',
                    'FG3M_away','FG3A_away','FG3_PCT_away','AST_away','REB_away','OREB_away','DREB_away',
                    'STL_away','BLK_away','TO_away','PF_away','PLUS_MINUS_away','STL/TO_away','EFG%_away',
                    'PPS_away','FIC_away']
# key columns
CURRENT_LINE_MAPPING_COLUMNS = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0]

# avg82 columns
AVG_82_TEAM_MAPPING_COLUMNS = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
                                1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
                                1, 0, 0, 0, 0]

# avg41 columns
AVG_41_MAPPING_COLUMNS = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
                            1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1,
                            1, 1, 1, 0, 1]

# LAG123
LAG123_MAPPING_COLUMNS = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1,
                          1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1]
LAG123_Intercept_COLUMNS = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                            0, 0, 0, 0, 0,
                         0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                            0, 0, 0, 0, 0]

# A vs B
AvsB12_MAPPING_COLUMN = [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0,
                            1, 1, 0, 1, 1,
                         1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0,
                            1, 1, 0, 1, 1]
AvsB12_Intercept_COLUMN = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0,
                         0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0,
                            0, 0, 0, 0, 0]



def generate_an_observation(currentLineIndex, home_back_82_List, away_back_82_List, home_away_back_2_List, lines):
    currentLine = []

    # 0. current line columns
    current_a_line = lines[currentLineIndex]
    for index, flag in enumerate(CURRENT_LINE_MAPPING_COLUMNS):
        if flag == 1:
            currentLine.append(current_a_line[index])

    # 1. past 2 HOME vs AWAY
    for j in range(0, HOME_AWAY_BACK_2):
        HAB2L = lines[int(home_away_back_2_List[j][0])]
        if home_away_back_2_List[j][1] == '1':
            for i, flag in enumerate(AvsB12_MAPPING_COLUMN):
                if flag == 1:
                    currentLine.append(HAB2L[i])
            for i, flag in enumerate(AvsB12_Intercept_COLUMN):
                if flag == 1:
                    currentLine.append(str(float(HAB2L[i])))
        elif home_away_back_2_List[j][1] == '0':
            currentLine += [str(-float(HAB2L[4])),HAB2L[31],HAB2L[35],HAB2L[36],HAB2L[37],HAB2L[41],
                                HAB2L[43],HAB2L[44],HAB2L[45],HAB2L[48],HAB2L[49],HAB2L[50],HAB2L[52],
                                HAB2L[53],HAB2L[55],HAB2L[56],HAB2L[5],HAB2L[9],HAB2L[10],HAB2L[11],
                                HAB2L[15],HAB2L[17],HAB2L[18],HAB2L[21],HAB2L[22],HAB2L[23],HAB2L[24],
                                HAB2L[26],HAB2L[27],HAB2L[29],HAB2L[30],'0','0','0','0','0','0','0',
                                '0','0','0','0']
        else:
            print(f'there is no location information for {HAB2L[:4]}')

    # 2. past 3 Home's and Away's + past 3 Home's and Away's  with weighted teamLoc information
    for i in range(0, Back_3):
        HLBD3 = lines[int(home_back_82_List[i][0])]
        # past home's home games
        if home_back_82_List[i][1] == '1':
            # past homes' home
            for index, flag in enumerate(LAG123_MAPPING_COLUMNS):
                if flag == 1:
                    currentLine.append(HLBD3[index])
            # past homes' home with weighted teamLoc information
            for index, flag in enumerate(LAG123_Intercept_COLUMNS):
                if flag == 1:
                    currentLine.append(str(float(HLBD3[index])))  # aLine_team_back_data[5] = teamLoc
        # past home's away games
        else:
            currentLine += [str(-float(HLBD3[4])),HLBD3[31],HLBD3[32],HLBD3[33],HLBD3[35],HLBD3[36],
                                   HLBD3[37],HLBD3[38],HLBD3[39],HLBD3[40],HLBD3[43],HLBD3[44],HLBD3[45],
                                HLBD3[46],HLBD3[47],HLBD3[48],HLBD3[49],HLBD3[50],HLBD3[51],HLBD3[52],HLBD3[53],
                                HLBD3[54],HLBD3[55],HLBD3[56],HLBD3[5],HLBD3[6],HLBD3[7],HLBD3[9],HLBD3[10],
                                HLBD3[11],HLBD3[12],HLBD3[13],HLBD3[14],HLBD3[17],HLBD3[18],HLBD3[19],HLBD3[20],
                                HLBD3[21],HLBD3[22],HLBD3[23],HLBD3[24],HLBD3[25],HLBD3[26],HLBD3[27],HLBD3[28],
                                HLBD3[29],HLBD3[30],'0','0','0','0','0','0','0']

        ALBD3 = lines[int(away_back_82_List[i][0])]
        # past away's home games
        if away_back_82_List[i][1] == '1':
            # past aways
            for index, flag in enumerate(LAG123_MAPPING_COLUMNS):
                if flag == 1:
                    currentLine.append(ALBD3[index])
            # past aways with weighted teamLoc information
            for index, flag in enumerate(LAG123_Intercept_COLUMNS):
                if flag == 1:
                    currentLine.append(str(float(ALBD3[index])))  # aLine_team_back_data[5] = teamLoc
        # past away's away games
        else:
            currentLine += [str(-float(ALBD3[4])), ALBD3[31], ALBD3[32], ALBD3[33], ALBD3[35], ALBD3[36],
                                ALBD3[37], ALBD3[38], ALBD3[39], ALBD3[40], ALBD3[43], ALBD3[44], ALBD3[45],
                                ALBD3[46], ALBD3[47], ALBD3[48], ALBD3[49], ALBD3[50], ALBD3[51], ALBD3[52], ALBD3[53],
                                ALBD3[54], ALBD3[55], ALBD3[56], ALBD3[5], ALBD3[6], ALBD3[7], ALBD3[9], ALBD3[10],
                                ALBD3[11], ALBD3[12], ALBD3[13], ALBD3[14], ALBD3[17], ALBD3[18], ALBD3[19], ALBD3[20],
                                ALBD3[21], ALBD3[22], ALBD3[23], ALBD3[24], ALBD3[25], ALBD3[26], ALBD3[27], ALBD3[28],
                                ALBD3[29], ALBD3[30], '0', '0', '0', '0', '0', '0', '0']

    # 3. past 82 averaged A's and B's
    HomeSamples = []
    AwaySamples = []
    for i in range(0, Back_82):
        home_newALine = []
        HB82 = lines[int(home_back_82_List[i][0])]
        if home_back_82_List[i][1] == '1':
            for index, flag in enumerate(AVG_82_TEAM_MAPPING_COLUMNS):
                if flag == 1:
                    home_newALine.append(HB82[index])
        else:
            home_newALine += [str(-float(HB82[4])),HB82[31],HB82[37],HB82[38],HB82[39],HB82[42],HB82[43],
                                  HB82[44],HB82[45],HB82[47],HB82[48],HB82[49],HB82[50],HB82[51],HB82[52],
                                  HB82[5],HB82[11],HB82[12],HB82[13],HB82[16],HB82[17],HB82[18],HB82[20],
                                  HB82[21],HB82[22],HB82[23],HB82[24],HB82[25],HB82[26]]
        HomeSamples.append(home_newALine)

        away_newALine = []
        AB82 = lines[int(away_back_82_List[i][0])]
        if away_back_82_List[i][1] == '1':
            for index, flag in enumerate(AVG_82_TEAM_MAPPING_COLUMNS):
                if flag == 1:
                    away_newALine.append(AB82[index])
        else:
            away_newALine += [str(-float(AB82[4])), AB82[31], AB82[37], AB82[38], AB82[39], AB82[42], AB82[43],
                                  AB82[44], AB82[45], AB82[47], AB82[48], AB82[49], AB82[50], AB82[51], AB82[52],
                                  AB82[5], AB82[11], AB82[12], AB82[13], AB82[16], AB82[17], AB82[18], AB82[20],
                                  AB82[21], AB82[22], AB82[23], AB82[24], AB82[25], AB82[26]]
        AwaySamples.append(away_newALine)

    team_averaged_ALine = calculateAverage(aBigList=HomeSamples)
    oppt_averaged_ALine = calculateAverage(aBigList=AwaySamples)
    currentLine = currentLine + team_averaged_ALine
    currentLine = currentLine + oppt_averaged_ALine

    # 4. past 41 averaged A's and B's
    HomeSamples = []
    AwaySamples = []
    for i in range(0, Back_41):
        home_newALine = []
        HB82 = lines[int(home_back_82_List[i][0])]
        if home_back_82_List[i][1] == '1':
            for index, flag in enumerate(AVG_41_MAPPING_COLUMNS):
                if flag == 1:
                    home_newALine.append(HB82[index])
        else:
            home_newALine += [str(-float(HB82[4])), HB82[31], HB82[32], HB82[33], HB82[37], HB82[38], HB82[39],
                              HB82[40], HB82[41], HB82[42], HB82[43], HB82[44], HB82[45], HB82[46], HB82[48],
                              HB82[49],HB82[51],HB82[52],HB82[53],HB82[54], HB82[56],
                              HB82[5], HB82[6], HB82[7], HB82[11], HB82[12], HB82[13], HB82[14], HB82[15],
                              HB82[16], HB82[17], HB82[18], HB82[19], HB82[21], HB82[22],HB82[23],HB82[25],
                              HB82[26],HB82[27],HB82[28],HB82[30]]
        HomeSamples.append(home_newALine)

        away_newALine = []
        AB82 = lines[int(away_back_82_List[i][0])]
        if away_back_82_List[i][1] == '1':
            for index, flag in enumerate(AVG_41_MAPPING_COLUMNS):
                if flag == 1:
                    away_newALine.append(AB82[index])
        else:
            away_newALine += [str(-float(AB82[4])), AB82[31], AB82[32], AB82[33], AB82[37], AB82[38], AB82[39],
                              AB82[40], AB82[41], AB82[42], AB82[43], AB82[44], AB82[45], AB82[46], AB82[48],
                              AB82[49],AB82[51],AB82[52],AB82[53],AB82[54], AB82[56],
                              AB82[5], AB82[6], AB82[7], AB82[11], AB82[12], AB82[13], AB82[14], AB82[15],
                              AB82[16], AB82[17], AB82[18], AB82[19], AB82[21], AB82[22],AB82[23],AB82[25],
                              AB82[26],AB82[27],AB82[28],AB82[30]]
        AwaySamples.append(away_newALine)

    team_averaged_ALine = calculateAverage(aBigList=HomeSamples)
    oppt_averaged_ALine = calculateAverage(aBigList=AwaySamples)
    currentLine = currentLine + team_averaged_ALine
    currentLine = currentLine + oppt_averaged_ALine

    return currentLine


def read_data():
    # colunms
    ## current line columns
    CurrentLineColumns = []
    for index, value in enumerate(CURRENT_LINE_MAPPING_COLUMNS):
        if value == 1:
            CurrentLineColumns.append(Original_COLUMNS[index])
    ## columns A vs B
    Columns_AvsB = []
    team_columns_AvsB = []
    for index, value in enumerate(AvsB12_MAPPING_COLUMN):
        if value == 1:
            team_columns_AvsB.append(Original_COLUMNS[index])
    for index, value in enumerate(AvsB12_Intercept_COLUMN):
        if value == 1:
            team_columns_AvsB.append('Loc*' + Original_COLUMNS[index])
    for time in range(2):
        for i in team_columns_AvsB:
            Columns_AvsB.append('AvsBLag' + str(time + 1) + i)
    ## past 3 A and B with teamLoc information columns
    TempColumns = []
    for index, value in enumerate(LAG123_MAPPING_COLUMNS):
        if value == 1:
            TempColumns.append(Original_COLUMNS[index])
    WeightedColumns = []
    for index, value in enumerate(LAG123_Intercept_COLUMNS):
        if value == 1:
            WeightedColumns.append('Loc*' + Original_COLUMNS[index])
    TempColumns.extend(WeightedColumns)
    TempColumnsA = []
    for v in TempColumns:
        TempColumnsA.append('H' + v)
    TempColumnsB = []
    for v in TempColumns:
        TempColumnsB.append('A' + v)
    TempColumnsA.extend(TempColumnsB)
    TempColumnsAB = TempColumnsA[:]
    PastThreeABColumns = []
    for i in range(1, 4):
        for v in TempColumnsAB:
            PastThreeABColumns.append(f"Lag{i}{v}")
    ##past 82 average and 41 average columns
    A82 = []
    for index, value in enumerate(AVG_82_TEAM_MAPPING_COLUMNS):
        if value == 1:
            A82.append(Original_COLUMNS[index])
    A82New = []
    for i in A82:
        A82New.append('Avg82home' + i)
    for i in A82:
        A82New.append('Avg82away' + i)
    A41 = []
    for index, value in enumerate(AVG_41_MAPPING_COLUMNS):
        if value == 1:
            A41.append(Original_COLUMNS[index])
    A41New = []
    for i in A41:
        A41New.append('Avg41home' + i)
    for i in A41:
        A41New.append('Avg41away' + i)
    ## add together
    NewColumns = CurrentLineColumns[:]
    NewColumns.extend(Columns_AvsB)
    NewColumns.extend(PastThreeABColumns)
    NewColumns.extend(A82New)
    NewColumns.extend(A41New)

    PlayerData = []
    with open('../data/games and player from 2004 to '
                  '2020/game_prediction_data/complete_0420_team_data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            PlayerData.append(row)
        lines = PlayerData[1:]

        bigTable = []
        bigTable.append(NewColumns)

    for j, line in enumerate(lines):
        home_ID = line[2]
        away_ID = line[3]
        game_ID = line[1]

        # 0:left side features, host position; 1:right side features, guest position
        home_back_82_List = []
        away_back_82_List = []
        home_away_back_2_List = []


        # loop all games before one game
        for i in range(j + 1, len(lines)-1):
            aPreLine = lines[i]
            # get each game's teamid and oppoid
            aPreHome_ID = aPreLine[2]
            aPreAway_ID = aPreLine[3]
            aPreGame_ID = aPreLine[1]
            # find 82 games' positions before one game with same homeid
            if home_ID == aPreHome_ID:
                home_back_82_List.append([str(i),'1'])
            elif home_ID == aPreAway_ID:
                home_back_82_List.append([str(i), '0'])
            # find 82 games' positions before one game with same awayid
            if away_ID == aPreHome_ID:
                away_back_82_List.append([str(i),'1'])
            elif away_ID == aPreAway_ID:
                away_back_82_List.append([str(i), '0'])
            # find last 2 games' positions before one game with same teamid and oppoid
            if home_ID == aPreHome_ID and away_ID == aPreAway_ID:
                home_away_back_2_List.append([str(i),'1'])
            elif home_ID == aPreAway_ID and away_ID == aPreHome_ID:
                home_away_back_2_List.append([str(i), '0'])
            # when all 82 teams' and oppos' positions and two games are find, stop finding.
            if len(home_back_82_List) >= Back_82 and len(away_back_82_List) >= Back_82 and len(
                    home_away_back_2_List) >= HOME_AWAY_BACK_2:
                break
            '''
            find the past 82, 10, 2 timeseries index
            generate a line of data(an obeservation/sample)
            '''
        # use position information above to find data for bigTable
        if len(home_back_82_List) >= Back_82 and len(away_back_82_List) >= Back_82 and len(
                home_away_back_2_List) >= HOME_AWAY_BACK_2:
            aSample = generate_an_observation(currentLineIndex=j,
                                              home_back_82_List=home_back_82_List,
                                              away_back_82_List=away_back_82_List,
                                              home_away_back_2_List=home_away_back_2_List,
                                              lines=lines)
            bigTable.append(aSample)

    with open('../data/games and player from 2004 to '
                  '2020/game_prediction_data/0420_with_back82_back41_back3_back2.csv', mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for new_data in bigTable:
            csv_writer.writerow(new_data)

        # outputfilename = "./12-18-with-back82-back10-back2.csv"
        # np.savetxt(outputfilename, np.array(bigTable))


def calculateAverage(aBigList):
    # 82 * 52 list[list[]]
    a = np.array(aBigList)
    a = a.astype(float)
    a = np.mean(a, axis=0)  # axis=0，计算每一列的均值
    a = a.tolist()
    a = list(map(str, a))
    return a


if __name__ == "__main__":
    read_data()
