import numpy as np

#parameters set up
RAW_FILE_PATH = '../data/game/12-18_standard_data.csv'
Back_82 = 82
Back_41 = 41
Back_10 = 10
TEAM_OPPT_BACK_2 = 2

#columns names set up
Original_COLUMNS = ['GAME_DATE_EST', 'teamAbbr', 'opptAbbr', 'GAME_ID', 'pointspread', 'teamLoc', 'teamRslt', 'teamMin',
           'teamDayOff', 'teamPTS', 'teamAST', 'teamTO(turnover)', 'teamSTL', 'teamBLK', 'teamPF(personal fouls)',
           'teamFGA', 'teamFGM', 'teamFG%', 'team2PA', 'team2PM', 'team2P%', 'team3PA', 'team3PM', 'team3P%', 'teamFTA',
           'teamFTM', 'teamFT%', 'teamORB', 'teamDRB', 'teamTRB', 'teamPTS1', 'teamPTS2', 'teamPTS3', 'teamPTS4',
           'teamPTS5', 'teamPTS6', 'teamPTS7', 'teamPTS8', 'teamTREB%', 'teamASST%', 'teamTS%', 'teamE(effective)FG%',
           'teamOREB%', 'teamDREB%', 'teamTO(turnover)%', 'teamSTL%', 'teamBLK%', 'teamBLKR',
           'teamPPS(points per shot)', 'teamFIC(floor impact counter)', 'teamFIC40(per 40 minuts)',
           'teamOrtg(offensive rating per 100 possesion)', 'teamDrtg(defensive rating per 100 possesion)',
           'teamEDiff(efficiency difference)', 'teamPlay%', 'teamAR', 'teamAST/TO', 'teamSTL/TO', 'opptConf', 'opptDiv',
           'opptLoc', 'opptRslt', 'opptMin', 'opptDayOff', 'opptPTS', 'opptAST', 'opptTO', 'opptSTL', 'opptBLK',
           'opptPF', 'opptFGA', 'opptFGM', 'opptFG%', 'oppt2PA', 'oppt2PM', 'oppt2P%', 'oppt3PA', 'oppt3PM', 'oppt3P%',
           'opptFTA', 'opptFTM', 'opptFT%', 'opptORB', 'opptDRB', 'opptTRB', 'opptPTS1', 'opptPTS2', 'opptPTS3',
           'opptPTS4', 'opptPTS5', 'opptPTS6', 'opptPTS7', 'opptPTS8', 'opptTREB%', 'opptASST%', 'opptTS%', 'opptEFG%',
           'opptOREB%', 'opptDREB%', 'opptTO%', 'opptSTL%', 'opptBLK%', 'opptBLKR', 'opptPPS', 'opptFIC', 'opptFIC40',
           'opptOrtg', 'opptDrtg', 'opptEDiff', 'opptPlay%', 'opptAR', 'opptAST/TO', 'opptSTL/TO', 'possession', 'pace']
#82 or lag10 for A
LEFT_HALF_TEAM_MAPPING_COLUMNS = [0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#82 or lag10 for B
RIGHT_HALF_OPPT_MAPPING_COLUMNS = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                   1, 0, 0]
# LABEL_MAPPING_COLUMNS not used yet
LABEL_MAPPING_COLUMNS = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# A vs B
TEAM_OPPT_MAPPING_COLUMN = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
## columns names A vs B set up
Columns_AvsB = []
temp_columns_AvsB = []
for index, value in enumerate(TEAM_OPPT_MAPPING_COLUMN):
    if value == 1:
        temp_columns_AvsB.append(Original_COLUMNS[index])
for time in range(2):
    for i in temp_columns_AvsB:
        Columns_AvsB.append('AvsBLag' + str(time+1) + i)
## columns names for A last 10 games
Columns_A_Lag10 = []
Columns_B_Lag10 = []
temp_columns_A = []
temp_columns_B = []
for index, value in enumerate(LEFT_HALF_TEAM_MAPPING_COLUMNS):
    if value == 1:
        temp_columns_A.append(Original_COLUMNS[index])
for index, value in enumerate(RIGHT_HALF_OPPT_MAPPING_COLUMNS):
    if value == 1:
        temp_columns_B.append(Original_COLUMNS[index])
for time in range(10):
    for i in temp_columns_A:
        Columns_A_Lag10.append('ALag' + str(time+1) + i)
for time in range(10):
    for i in temp_columns_B:
        Columns_B_Lag10.append('BLag' + str(time+1) + i)
Columns_A_Lag10.extend(Columns_B_Lag10)
columns_lag10 = Columns_A_Lag10
##columns names for 82 average
columns_82_41 = []
for i in temp_columns_A:
    columns_82_41.append('Avg82A' + i)
for i in temp_columns_B:
    columns_82_41.append('Avg82B' + i)
for i in temp_columns_A:
    columns_82_41.append('Avg41A' + i)
for i in temp_columns_B:
    columns_82_41.append('Avg41B' + i)
##merge all names together
Original_COLUMNS.extend(Columns_AvsB)
Original_COLUMNS.extend(columns_lag10)
Original_COLUMNS.extend(columns_82_41)
NewColumns = Original_COLUMNS



def read_data():
    with open(RAW_FILE_PATH, "r") as datafile:
        lines = datafile.readlines()
        bigTable = []
        bigTable.append(NewColumns)

        for j, line in enumerate(lines):
            if j == 0: continue
            columns = line.split(',')
            team_ID = columns[1]
            oppt_ID = columns[2]
            # 0:left side features, host position; 1:right side features, guest position
            team_back_82_List = []
            oppt_back_82_List = []
            team_oppt_back_2_List = []

            if j > Back_82:
                #loop all games before one game
                for i in range(j - 1, 0, -1):
                    aPreLine = lines[i]
                    aPreLineColumns = aPreLine.split(',')
                    #get each game's teamid and oppoid
                    aPreTeam_ID = aPreLineColumns[1]
                    aPreOppt_ID = aPreLineColumns[2]
                    #find 82 games' positions before one game with same teamid
                    if team_ID == aPreTeam_ID:
                        team_back_82_List.append(str(i) + "_" + str(0))
                    #find 82 games' positions before one game with same oppoid
                    if oppt_ID == aPreOppt_ID:
                        oppt_back_82_List.append(str(i) + "_" + str(1))
                    #find last 2 games' positions before one game with same teamid and oppoid
                    if team_ID == aPreTeam_ID and oppt_ID == aPreOppt_ID:
                        team_oppt_back_2_List.append(str(i))
                    #when all 82 teams' and oppos' positions and two games are find, stop finding.
                    if len(team_back_82_List) >= Back_82 and len(oppt_back_82_List) >= Back_82 and len(
                            team_oppt_back_2_List) >= TEAM_OPPT_BACK_2:
                        break
                    '''
                    find the past 82, 10, 2 timeseries index
                    generate a line of data(an obeservation/sample)
                    '''
                #use position information above to find data for bigTable
                if len(team_back_82_List) >= Back_82 and len(oppt_back_82_List) >= Back_82 and len(
                        team_oppt_back_2_List) >= TEAM_OPPT_BACK_2:
                    aSample = generate_an_observation(currentLineIndex=j,
                                                      team_back_82_List=team_back_82_List,
                                                      oppt_back_82_List=oppt_back_82_List,
                                                      team_oppt_back_2_List=team_oppt_back_2_List,
                                                      lines=lines)
                    bigTable.append(aSample)


        outputfilename = "../data/game/12-18-with-back82-back10-back2.csv"
        with open(outputfilename, 'w') as fileObject:
            for aNewSample in bigTable:
                for aNewColumn in aNewSample:
                    aNewColumn = aNewColumn.replace('\n', "")
                    fileObject.write(aNewColumn)
                    fileObject.write(",")
                fileObject.write('\n')
            fileObject.close()

        # outputfilename = "./12-18-with-back82-back10-back2.csv"
        # np.savetxt(outputfilename, np.array(bigTable))


def generate_an_observation(currentLineIndex, team_back_82_List, oppt_back_82_List, team_oppt_back_2_List, lines):
    currentLine = lines[currentLineIndex]
    currentLine = currentLine.split(',')

    # past 2 AvsB
    for j in range(0, TEAM_OPPT_BACK_2):
        team_oppt_back_2_aLine = lines[int(team_oppt_back_2_List[j])]
        team_oppt_back_2_aLine_columns = team_oppt_back_2_aLine.split(',')
        for i, flag in enumerate(TEAM_OPPT_MAPPING_COLUMN):
            if flag == 1:
                currentLine.append(team_oppt_back_2_aLine_columns[i])

    # past 10 A's and B's
    for i in range(0, Back_10):
        team_back_10_aLine = team_back_82_List[i]
        team_back_10_aLine_columns = team_back_10_aLine.split('_')
        aLine_team_back_10_data = lines[int(team_back_10_aLine_columns[0])]
        aLine_team_back_10_data = aLine_team_back_10_data.split(',')
        if int(team_back_10_aLine_columns[1]) == 0:
            for index, flag in enumerate(LEFT_HALF_TEAM_MAPPING_COLUMNS):
                if flag == 1:
                    currentLine.append(aLine_team_back_10_data[index])
        elif int(team_back_10_aLine_columns[1]) == 1:
            for index, flag in enumerate(RIGHT_HALF_OPPT_MAPPING_COLUMNS):
                if flag == 1:
                    currentLine.append(aLine_team_back_10_data[index])

        oppt_back_10_aLine = oppt_back_82_List[i]
        oppt_back_10_aLine_columns = oppt_back_10_aLine.split('_')
        aLine_oppt_back_10_data = lines[int(oppt_back_10_aLine_columns[0])]
        aLine_oppt_back_10_data = aLine_oppt_back_10_data.split(',')
        if int(oppt_back_10_aLine_columns[1]) == 0:
            for index, flag in enumerate(LEFT_HALF_TEAM_MAPPING_COLUMNS):
                if flag == 1:
                    currentLine.append(aLine_oppt_back_10_data[index])
        elif int(oppt_back_10_aLine_columns[1]) == 1:
            for index, flag in enumerate(RIGHT_HALF_OPPT_MAPPING_COLUMNS):
                if flag == 1:
                    currentLine.append(aLine_oppt_back_10_data[index])

    # past 82 averaged A's and B's
    TeamSamples = []
    OpptSamples = []
    for i in range(0, Back_82):
        team_newALine = []
        team_back_82_aLine = team_back_82_List[i]
        team_back_82_aLine_columns = team_back_82_aLine.split('_')
        aLine_team_back_82_data = lines[int(team_back_82_aLine_columns[0])]
        aLine_team_back_82_data = aLine_team_back_82_data.split(',')
        if int(team_back_82_aLine_columns[1]) == 0:
            for index, flag in enumerate(LEFT_HALF_TEAM_MAPPING_COLUMNS):
                if flag == 1:
                    team_newALine.append(aLine_team_back_82_data[index])
        elif int(team_back_82_aLine_columns[1]) == 1:
            for index, flag in enumerate(RIGHT_HALF_OPPT_MAPPING_COLUMNS):
                if flag == 1:
                    team_newALine.append(aLine_team_back_82_data[index])
        TeamSamples.append(team_newALine)

        oppt_newALine = []
        oppt_back_82_aLine = oppt_back_82_List[i]
        oppt_back_82_aLine_columns = oppt_back_82_aLine.split('_')
        aLine_oppt_back_82_data = lines[int(oppt_back_82_aLine_columns[0])]
        aLine_oppt_back_82_data = aLine_oppt_back_82_data.split(',')
        if int(oppt_back_82_aLine_columns[1]) == 0:
            for index, flag in enumerate(LEFT_HALF_TEAM_MAPPING_COLUMNS):
                if flag == 1:
                    oppt_newALine.append(aLine_oppt_back_82_data[index])
        elif int(oppt_back_82_aLine_columns[1]) == 1:
            for index, flag in enumerate(RIGHT_HALF_OPPT_MAPPING_COLUMNS):
                if flag == 1:
                    oppt_newALine.append(aLine_oppt_back_82_data[index])
        OpptSamples.append(oppt_newALine)

    team_averaged_ALine = calculateAverage(aBigList=TeamSamples)
    oppt_averaged_ALine = calculateAverage(aBigList=OpptSamples)
    currentLine = currentLine + team_averaged_ALine
    currentLine = currentLine + oppt_averaged_ALine
    
    # past 41 averaged A's and B's
    TeamSamples = []
    OpptSamples = []
    for i in range(0, Back_41):
        team_newALine = []
        team_back_82_aLine = team_back_82_List[i]
        team_back_82_aLine_columns = team_back_82_aLine.split('_')
        aLine_team_back_82_data = lines[int(team_back_82_aLine_columns[0])]
        aLine_team_back_82_data = aLine_team_back_82_data.split(',')
        if int(team_back_82_aLine_columns[1]) == 0:
            for index, flag in enumerate(LEFT_HALF_TEAM_MAPPING_COLUMNS):
                if flag == 1:
                    team_newALine.append(aLine_team_back_82_data[index])
        elif int(team_back_82_aLine_columns[1]) == 1:
            for index, flag in enumerate(RIGHT_HALF_OPPT_MAPPING_COLUMNS):
                if flag == 1:
                    team_newALine.append(aLine_team_back_82_data[index])
        TeamSamples.append(team_newALine)

        oppt_newALine = []
        oppt_back_82_aLine = oppt_back_82_List[i]
        oppt_back_82_aLine_columns = oppt_back_82_aLine.split('_')
        aLine_oppt_back_82_data = lines[int(oppt_back_82_aLine_columns[0])]
        aLine_oppt_back_82_data = aLine_oppt_back_82_data.split(',')
        if int(oppt_back_82_aLine_columns[1]) == 0:
            for index, flag in enumerate(LEFT_HALF_TEAM_MAPPING_COLUMNS):
                if flag == 1:
                    oppt_newALine.append(aLine_oppt_back_82_data[index])
        elif int(oppt_back_82_aLine_columns[1]) == 1:
            for index, flag in enumerate(RIGHT_HALF_OPPT_MAPPING_COLUMNS):
                if flag == 1:
                    oppt_newALine.append(aLine_oppt_back_82_data[index])
        OpptSamples.append(oppt_newALine)

    team_averaged_ALine = calculateAverage(aBigList=TeamSamples)
    oppt_averaged_ALine = calculateAverage(aBigList=OpptSamples)
    currentLine = currentLine + team_averaged_ALine
    currentLine = currentLine + oppt_averaged_ALine
    return currentLine


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
