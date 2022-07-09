# data preparation
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('../data/game/12-18-with-back82-back10-back2.csv')
feature_cols = ['teamLoc', 'teamDayOff', 'teamLocAvsBLag1', 'pointspreadAvsBLag1', 'teamLocAvsBLag2',
                'pointspreadAvsBLag2', 'teamLocAvsBLag3', 'pointspreadAvsBLag3', 'teamLocAvsBLag4',
                'pointspreadAvsBLag4', 'teamLocAvsBLag5', 'pointspreadAvsBLag5', 'teamLocAvsBLag6',
                'pointspreadAvsBLag6', 'teamLocAvsBLag7', 'pointspreadAvsBLag7', 'teamLocAvsBLag8',
                'pointspreadAvsBLag8', 'teamLocAvsBLag9', 'pointspreadAvsBLag9', 'teamLocAvsBLag10',
                'pointspreadAvsBLag10', 'pointspreadALag10', 'teamLocALag10', 'teamMinALag10', 'teamDayOffALag10',
                'teamPTSALag10', 'teamASTALag10', 'teamTOALag10', 'teamSTLALag10', 'teamBLKALag10', 'teamPFALag10',
                'teamFGAALag10', 'teamFGMALag10', 'teamFG%ALag10', 'team2PAALag10', 'team2PMALag10', 'team2P%ALag10',
                'team3PAALag10', 'team3PMALag10', 'team3P%ALag10', 'teamFTAALag10', 'teamFTMALag10', 'teamFT%ALag10',
                'teamORBALag10', 'teamDRBALag10', 'teamTRBALag10', 'teamPTS1ALag10', 'teamPTS2ALag10', 'teamPTS3ALag10',
                'teamPTS4ALag10', 'teamPTS5ALag10', 'teamPTS6ALag10', 'teamPTS7ALag10', 'teamPTS8ALag10',
                'teamTREB%ALag10', 'teamASST%ALag10', 'teamTS%ALag10', 'teamE(effective)FG%ALag10', 'teamOREB%ALag10',
                'teamDREB%ALag10', 'teamTO%ALag10', 'teamSTL%ALag10', 'teamBLK%ALag10', 'teamBLKRALag10',
                'teamPPSALag10', 'teamFICALag10', 'teamFIC40ALag10', 'teamOrtgALag10', 'teamDrtgALag10',
                'teamEDiffALag10', 'teamPlay%ALag10', 'teamARALag10', 'teamAST/TOALag10', 'teamSTL/TOALag10',
                'pointspreadALag9', 'teamLocALag9', 'teamMinALag9', 'teamDayOffALag9', 'teamPTSALag9', 'teamASTALag9',
                'teamTOALag9', 'teamSTLALag9', 'teamBLKALag9', 'teamPFALag9', 'teamFGAALag9', 'teamFGMALag9',
                'teamFG%ALag9', 'team2PAALag9', 'team2PMALag9', 'team2P%ALag9', 'team3PAALag9', 'team3PMALag9',
                'team3P%ALag9', 'teamFTAALag9', 'teamFTMALag9', 'teamFT%ALag9', 'teamORBALag9', 'teamDRBALag9',
                'teamTRBALag9', 'teamPTS1ALag9', 'teamPTS2ALag9', 'teamPTS3ALag9', 'teamPTS4ALag9', 'teamPTS5ALag9',
                'teamPTS6ALag9', 'teamPTS7ALag9', 'teamPTS8ALag9', 'teamTREB%ALag9', 'teamASST%ALag9', 'teamTS%ALag9',
                'teamE(effective)FG%ALag9', 'teamOREB%ALag9', 'teamDREB%ALag9', 'teamTO%ALag9', 'teamSTL%ALag9',
                'teamBLK%ALag9', 'teamBLKRALag9', 'teamPPSALag9', 'teamFICALag9', 'teamFIC40ALag9', 'teamOrtgALag9',
                'teamDrtgALag9', 'teamEDiffALag9', 'teamPlay%ALag9', 'teamARALag9', 'teamAST/TOALag9',
                'teamSTL/TOALag9', 'pointspreadALag8', 'teamLocALag8', 'teamMinALag8', 'teamDayOffALag8',
                'teamPTSALag8', 'teamASTALag8', 'teamTOALag8', 'teamSTLALag8', 'teamBLKALag8', 'teamPFALag8',
                'teamFGAALag8', 'teamFGMALag8', 'teamFG%ALag8', 'team2PAALag8', 'team2PMALag8', 'team2P%ALag8',
                'team3PAALag8', 'team3PMALag8', 'team3P%ALag8', 'teamFTAALag8', 'teamFTMALag8', 'teamFT%ALag8',
                'teamORBALag8', 'teamDRBALag8', 'teamTRBALag8', 'teamPTS1ALag8', 'teamPTS2ALag8', 'teamPTS3ALag8',
                'teamPTS4ALag8', 'teamPTS5ALag8', 'teamPTS6ALag8', 'teamPTS7ALag8', 'teamPTS8ALag8', 'teamTREB%ALag8',
                'teamASST%ALag8', 'teamTS%ALag8', 'teamE(effective)FG%ALag8', 'teamOREB%ALag8', 'teamDREB%ALag8',
                'teamTO%ALag8', 'teamSTL%ALag8', 'teamBLK%ALag8', 'teamBLKRALag8', 'teamPPSALag8', 'teamFICALag8',
                'teamFIC40ALag8', 'teamOrtgALag8', 'teamDrtgALag8', 'teamEDiffALag8', 'teamPlay%ALag8', 'teamARALag8',
                'teamAST/TOALag8', 'teamSTL/TOALag8', 'pointspreadALag7', 'teamLocALag7', 'teamMinALag7',
                'teamDayOffALag7', 'teamPTSALag7', 'teamASTALag7', 'teamTOALag7', 'teamSTLALag7', 'teamBLKALag7',
                'teamPFALag7', 'teamFGAALag7', 'teamFGMALag7', 'teamFG%ALag7', 'team2PAALag7', 'team2PMALag7',
                'team2P%ALag7', 'team3PAALag7', 'team3PMALag7', 'team3P%ALag7', 'teamFTAALag7', 'teamFTMALag7',
                'teamFT%ALag7', 'teamORBALag7', 'teamDRBALag7', 'teamTRBALag7', 'teamPTS1ALag7', 'teamPTS2ALag7',
                'teamPTS3ALag7', 'teamPTS4ALag7', 'teamPTS5ALag7', 'teamPTS6ALag7', 'teamPTS7ALag7', 'teamPTS8ALag7',
                'teamTREB%ALag7', 'teamASST%ALag7', 'teamTS%ALag7', 'teamE(effective)FG%ALag7', 'teamOREB%ALag7',
                'teamDREB%ALag7', 'teamTO%ALag7', 'teamSTL%ALag7', 'teamBLK%ALag7', 'teamBLKRALag7', 'teamPPSALag7',
                'teamFICALag7', 'teamFIC40ALag7', 'teamOrtgALag7', 'teamDrtgALag7', 'teamEDiffALag7', 'teamPlay%ALag7',
                'teamARALag7', 'teamAST/TOALag7', 'teamSTL/TOALag7', 'pointspreadALag6', 'teamLocALag6', 'teamMinALag6',
                'teamDayOffALag6', 'teamPTSALag6', 'teamASTALag6', 'teamTOALag6', 'teamSTLALag6', 'teamBLKALag6',
                'teamPFALag6', 'teamFGAALag6', 'teamFGMALag6', 'teamFG%ALag6', 'team2PAALag6', 'team2PMALag6',
                'team2P%ALag6', 'team3PAALag6', 'team3PMALag6', 'team3P%ALag6', 'teamFTAALag6', 'teamFTMALag6',
                'teamFT%ALag6', 'teamORBALag6', 'teamDRBALag6', 'teamTRBALag6', 'teamPTS1ALag6', 'teamPTS2ALag6',
                'teamPTS3ALag6', 'teamPTS4ALag6', 'teamPTS5ALag6', 'teamPTS6ALag6', 'teamPTS7ALag6', 'teamPTS8ALag6',
                'teamTREB%ALag6', 'teamASST%ALag6', 'teamTS%ALag6', 'teamE(effective)FG%ALag6', 'teamOREB%ALag6',
                'teamDREB%ALag6', 'teamTO%ALag6', 'teamSTL%ALag6', 'teamBLK%ALag6', 'teamBLKRALag6', 'teamPPSALag6',
                'teamFICALag6', 'teamFIC40ALag6', 'teamOrtgALag6', 'teamDrtgALag6', 'teamEDiffALag6', 'teamPlay%ALag6',
                'teamARALag6', 'teamAST/TOALag6', 'teamSTL/TOALag6', 'pointspreadALag5', 'teamLocALag5', 'teamMinALag5',
                'teamDayOffALag5', 'teamPTSALag5', 'teamASTALag5', 'teamTOALag5', 'teamSTLALag5', 'teamBLKALag5',
                'teamPFALag5', 'teamFGAALag5', 'teamFGMALag5', 'teamFG%ALag5', 'team2PAALag5', 'team2PMALag5',
                'team2P%ALag5', 'team3PAALag5', 'team3PMALag5', 'team3P%ALag5', 'teamFTAALag5', 'teamFTMALag5',
                'teamFT%ALag5', 'teamORBALag5', 'teamDRBALag5', 'teamTRBALag5', 'teamPTS1ALag5', 'teamPTS2ALag5',
                'teamPTS3ALag5', 'teamPTS4ALag5', 'teamPTS5ALag5', 'teamPTS6ALag5', 'teamPTS7ALag5', 'teamPTS8ALag5',
                'teamTREB%ALag5', 'teamASST%ALag5', 'teamTS%ALag5', 'teamE(effective)FG%ALag5', 'teamOREB%ALag5',
                'teamDREB%ALag5', 'teamTO%ALag5', 'teamSTL%ALag5', 'teamBLK%ALag5', 'teamBLKRALag5', 'teamPPSALag5',
                'teamFICALag5', 'teamFIC40ALag5', 'teamOrtgALag5', 'teamDrtgALag5', 'teamEDiffALag5', 'teamPlay%ALag5',
                'teamARALag5', 'teamAST/TOALag5', 'teamSTL/TOALag5', 'pointspreadALag4', 'teamLocALag4', 'teamMinALag4',
                'teamDayOffALag4', 'teamPTSALag4', 'teamASTALag4', 'teamTOALag4', 'teamSTLALag4', 'teamBLKALag4',
                'teamPFALag4', 'teamFGAALag4', 'teamFGMALag4', 'teamFG%ALag4', 'team2PAALag4', 'team2PMALag4',
                'team2P%ALag4', 'team3PAALag4', 'team3PMALag4', 'team3P%ALag4', 'teamFTAALag4', 'teamFTMALag4',
                'teamFT%ALag4', 'teamORBALag4', 'teamDRBALag4', 'teamTRBALag4', 'teamPTS1ALag4', 'teamPTS2ALag4',
                'teamPTS3ALag4', 'teamPTS4ALag4', 'teamPTS5ALag4', 'teamPTS6ALag4', 'teamPTS7ALag4', 'teamPTS8ALag4',
                'teamTREB%ALag4', 'teamASST%ALag4', 'teamTS%ALag4', 'teamE(effective)FG%ALag4', 'teamOREB%ALag4',
                'teamDREB%ALag4', 'teamTO%ALag4', 'teamSTL%ALag4', 'teamBLK%ALag4', 'teamBLKRALag4', 'teamPPSALag4',
                'teamFICALag4', 'teamFIC40ALag4', 'teamOrtgALag4', 'teamDrtgALag4', 'teamEDiffALag4', 'teamPlay%ALag4',
                'teamARALag4', 'teamAST/TOALag4', 'teamSTL/TOALag4', 'pointspreadALag3', 'teamLocALag3', 'teamMinALag3',
                'teamDayOffALag3', 'teamPTSALag3', 'teamASTALag3', 'teamTOALag3', 'teamSTLALag3', 'teamBLKALag3',
                'teamPFALag3', 'teamFGAALag3', 'teamFGMALag3', 'teamFG%ALag3', 'team2PAALag3', 'team2PMALag3',
                'team2P%ALag3', 'team3PAALag3', 'team3PMALag3', 'team3P%ALag3', 'teamFTAALag3', 'teamFTMALag3',
                'teamFT%ALag3', 'teamORBALag3', 'teamDRBALag3', 'teamTRBALag3', 'teamPTS1ALag3', 'teamPTS2ALag3',
                'teamPTS3ALag3', 'teamPTS4ALag3', 'teamPTS5ALag3', 'teamPTS6ALag3', 'teamPTS7ALag3', 'teamPTS8ALag3',
                'teamTREB%ALag3', 'teamASST%ALag3', 'teamTS%ALag3', 'teamE(effective)FG%ALag3', 'teamOREB%ALag3',
                'teamDREB%ALag3', 'teamTO%ALag3', 'teamSTL%ALag3', 'teamBLK%ALag3', 'teamBLKRALag3', 'teamPPSALag3',
                'teamFICALag3', 'teamFIC40ALag3', 'teamOrtgALag3', 'teamDrtgALag3', 'teamEDiffALag3', 'teamPlay%ALag3',
                'teamARALag3', 'teamAST/TOALag3', 'teamSTL/TOALag3', 'pointspreadALag2', 'teamLocALag2', 'teamMinALag2',
                'teamDayOffALag2', 'teamPTSALag2', 'teamASTALag2', 'teamTOALag2', 'teamSTLALag2', 'teamBLKALag2',
                'teamPFALag2', 'teamFGAALag2', 'teamFGMALag2', 'teamFG%ALag2', 'team2PAALag2', 'team2PMALag2',
                'team2P%ALag2', 'team3PAALag2', 'team3PMALag2', 'team3P%ALag2', 'teamFTAALag2', 'teamFTMALag2',
                'teamFT%ALag2', 'teamORBALag2', 'teamDRBALag2', 'teamTRBALag2', 'teamPTS1ALag2', 'teamPTS2ALag2',
                'teamPTS3ALag2', 'teamPTS4ALag2', 'teamPTS5ALag2', 'teamPTS6ALag2', 'teamPTS7ALag2', 'teamPTS8ALag2',
                'teamTREB%ALag2', 'teamASST%ALag2', 'teamTS%ALag2', 'teamE(effective)FG%ALag2', 'teamOREB%ALag2',
                'teamDREB%ALag2', 'teamTO%ALag2', 'teamSTL%ALag2', 'teamBLK%ALag2', 'teamBLKRALag2', 'teamPPSALag2',
                'teamFICALag2', 'teamFIC40ALag2', 'teamOrtgALag2', 'teamDrtgALag2', 'teamEDiffALag2', 'teamPlay%ALag2',
                'teamARALag2', 'teamAST/TOALag2', 'teamSTL/TOALag2', 'pointspreadALag1', 'teamLocALag1', 'teamMinALag1',
                'teamDayOffALag1', 'teamPTSALag1', 'teamASTALag1', 'teamTOALag1', 'teamSTLALag1', 'teamBLKALag1',
                'teamPFALag1', 'teamFGAALag1', 'teamFGMALag1', 'teamFG%ALag1', 'team2PAALag1', 'team2PMALag1',
                'team2P%ALag1', 'team3PAALag1', 'team3PMALag1', 'team3P%ALag1', 'teamFTAALag1', 'teamFTMALag1',
                'teamFT%ALag1', 'teamORBALag1', 'teamDRBALag1', 'teamTRBALag1', 'teamPTS1ALag1', 'teamPTS2ALag1',
                'teamPTS3ALag1', 'teamPTS4ALag1', 'teamPTS5ALag1', 'teamPTS6ALag1', 'teamPTS7ALag1', 'teamPTS8ALag1',
                'teamTREB%ALag1', 'teamASST%ALag1', 'teamTS%ALag1', 'teamE(effective)FG%ALag1', 'teamOREB%ALag1',
                'teamDREB%ALag1', 'teamTO%ALag1', 'teamSTL%ALag1', 'teamBLK%ALag1', 'teamBLKRALag1', 'teamPPSALag1',
                'teamFICALag1', 'teamFIC40ALag1', 'teamOrtgALag1', 'teamDrtgALag1', 'teamEDiffALag1', 'teamPlay%ALag1',
                'teamARALag1', 'teamAST/TOALag1', 'teamSTL/TOALag1', 'pointspreadBLag10', 'teamLocBLag10',
                'teamMinBLag10', 'teamDayOffBLag10', 'teamPTSBLag10', 'teamASTBLag10', 'teamTOBLag10', 'teamSTLBLag10',
                'teamBLKBLag10', 'teamPFBLag10', 'teamFGABLag10', 'teamFGMBLag10', 'teamFG%BLag10', 'team2PABLag10',
                'team2PMBLag10', 'team2P%BLag10', 'team3PABLag10', 'team3PMBLag10', 'team3P%BLag10', 'teamFTABLag10',
                'teamFTMBLag10', 'teamFT%BLag10', 'teamORBBLag10', 'teamDRBBLag10', 'teamTRBBLag10', 'teamPTS1BLag10',
                'teamPTS2BLag10', 'teamPTS3BLag10', 'teamPTS4BLag10', 'teamPTS5BLag10', 'teamPTS6BLag10',
                'teamPTS7BLag10', 'teamPTS8BLag10', 'teamTREB%BLag10', 'teamASST%BLag10', 'teamTS%BLag10',
                'teamE(effective)FG%BLag10', 'teamOREB%BLag10', 'teamDREB%BLag10', 'teamTO%BLag10', 'teamSTL%BLag10',
                'teamBLK%BLag10', 'teamBLKRBLag10', 'teamPPSBLag10', 'teamFICBLag10', 'teamFIC40BLag10',
                'teamOrtgBLag10', 'teamDrtgBLag10', 'teamEDiffBLag10', 'teamPlay%BLag10', 'teamARBLag10',
                'teamAST/TOBLag10', 'teamSTL/TOBLag10', 'pointspreadBLag9', 'teamLocBLag9', 'teamMinBLag9',
                'teamDayOffBLag9', 'teamPTSBLag9', 'teamASTBLag9', 'teamTOBLag9', 'teamSTLBLag9', 'teamBLKBLag9',
                'teamPFBLag9', 'teamFGABLag9', 'teamFGMBLag9', 'teamFG%BLag9', 'team2PABLag9', 'team2PMBLag9',
                'team2P%BLag9', 'team3PABLag9', 'team3PMBLag9', 'team3P%BLag9', 'teamFTABLag9', 'teamFTMBLag9',
                'teamFT%BLag9', 'teamORBBLag9', 'teamDRBBLag9', 'teamTRBBLag9', 'teamPTS1BLag9', 'teamPTS2BLag9',
                'teamPTS3BLag9', 'teamPTS4BLag9', 'teamPTS5BLag9', 'teamPTS6BLag9', 'teamPTS7BLag9', 'teamPTS8BLag9',
                'teamTREB%BLag9', 'teamASST%BLag9', 'teamTS%BLag9', 'teamE(effective)FG%BLag9', 'teamOREB%BLag9',
                'teamDREB%BLag9', 'teamTO%BLag9', 'teamSTL%BLag9', 'teamBLK%BLag9', 'teamBLKRBLag9', 'teamPPSBLag9',
                'teamFICBLag9', 'teamFIC40BLag9', 'teamOrtgBLag9', 'teamDrtgBLag9', 'teamEDiffBLag9', 'teamPlay%BLag9',
                'teamARBLag9', 'teamAST/TOBLag9', 'teamSTL/TOBLag9', 'pointspreadBLag8', 'teamLocBLag8', 'teamMinBLag8',
                'teamDayOffBLag8', 'teamPTSBLag8', 'teamASTBLag8', 'teamTOBLag8', 'teamSTLBLag8', 'teamBLKBLag8',
                'teamPFBLag8', 'teamFGABLag8', 'teamFGMBLag8', 'teamFG%BLag8', 'team2PABLag8', 'team2PMBLag8',
                'team2P%BLag8', 'team3PABLag8', 'team3PMBLag8', 'team3P%BLag8', 'teamFTABLag8', 'teamFTMBLag8',
                'teamFT%BLag8', 'teamORBBLag8', 'teamDRBBLag8', 'teamTRBBLag8', 'teamPTS1BLag8', 'teamPTS2BLag8',
                'teamPTS3BLag8', 'teamPTS4BLag8', 'teamPTS5BLag8', 'teamPTS6BLag8', 'teamPTS7BLag8', 'teamPTS8BLag8',
                'teamTREB%BLag8', 'teamASST%BLag8', 'teamTS%BLag8', 'teamE(effective)FG%BLag8', 'teamOREB%BLag8',
                'teamDREB%BLag8', 'teamTO%BLag8', 'teamSTL%BLag8', 'teamBLK%BLag8', 'teamBLKRBLag8', 'teamPPSBLag8',
                'teamFICBLag8', 'teamFIC40BLag8', 'teamOrtgBLag8', 'teamDrtgBLag8', 'teamEDiffBLag8', 'teamPlay%BLag8',
                'teamARBLag8', 'teamAST/TOBLag8', 'teamSTL/TOBLag8', 'pointspreadBLag7', 'teamLocBLag7', 'teamMinBLag7',
                'teamDayOffBLag7', 'teamPTSBLag7', 'teamASTBLag7', 'teamTOBLag7', 'teamSTLBLag7', 'teamBLKBLag7',
                'teamPFBLag7', 'teamFGABLag7', 'teamFGMBLag7', 'teamFG%BLag7', 'team2PABLag7', 'team2PMBLag7',
                'team2P%BLag7', 'team3PABLag7', 'team3PMBLag7', 'team3P%BLag7', 'teamFTABLag7', 'teamFTMBLag7',
                'teamFT%BLag7', 'teamORBBLag7', 'teamDRBBLag7', 'teamTRBBLag7', 'teamPTS1BLag7', 'teamPTS2BLag7',
                'teamPTS3BLag7', 'teamPTS4BLag7', 'teamPTS5BLag7', 'teamPTS6BLag7', 'teamPTS7BLag7', 'teamPTS8BLag7',
                'teamTREB%BLag7', 'teamASST%BLag7', 'teamTS%BLag7', 'teamE(effective)FG%BLag7', 'teamOREB%BLag7',
                'teamDREB%BLag7', 'teamTO%BLag7', 'teamSTL%BLag7', 'teamBLK%BLag7', 'teamBLKRBLag7', 'teamPPSBLag7',
                'teamFICBLag7', 'teamFIC40BLag7', 'teamOrtgBLag7', 'teamDrtgBLag7', 'teamEDiffBLag7', 'teamPlay%BLag7',
                'teamARBLag7', 'teamAST/TOBLag7', 'teamSTL/TOBLag7', 'pointspreadBLag6', 'teamLocBLag6', 'teamMinBLag6',
                'teamDayOffBLag6', 'teamPTSBLag6', 'teamASTBLag6', 'teamTOBLag6', 'teamSTLBLag6', 'teamBLKBLag6',
                'teamPFBLag6', 'teamFGABLag6', 'teamFGMBLag6', 'teamFG%BLag6', 'team2PABLag6', 'team2PMBLag6',
                'team2P%BLag6', 'team3PABLag6', 'team3PMBLag6', 'team3P%BLag6', 'teamFTABLag6', 'teamFTMBLag6',
                'teamFT%BLag6', 'teamORBBLag6', 'teamDRBBLag6', 'teamTRBBLag6', 'teamPTS1BLag6', 'teamPTS2BLag6',
                'teamPTS3BLag6', 'teamPTS4BLag6', 'teamPTS5BLag6', 'teamPTS6BLag6', 'teamPTS7BLag6', 'teamPTS8BLag6',
                'teamTREB%BLag6', 'teamASST%BLag6', 'teamTS%BLag6', 'teamE(effective)FG%BLag6', 'teamOREB%BLag6',
                'teamDREB%BLag6', 'teamTO%BLag6', 'teamSTL%BLag6', 'teamBLK%BLag6', 'teamBLKRBLag6', 'teamPPSBLag6',
                'teamFICBLag6', 'teamFIC40BLag6', 'teamOrtgBLag6', 'teamDrtgBLag6', 'teamEDiffBLag6', 'teamPlay%BLag6',
                'teamARBLag6', 'teamAST/TOBLag6', 'teamSTL/TOBLag6', 'pointspreadBLag5', 'teamLocBLag5', 'teamMinBLag5',
                'teamDayOffBLag5', 'teamPTSBLag5', 'teamASTBLag5', 'teamTOBLag5', 'teamSTLBLag5', 'teamBLKBLag5',
                'teamPFBLag5', 'teamFGABLag5', 'teamFGMBLag5', 'teamFG%BLag5', 'team2PABLag5', 'team2PMBLag5',
                'team2P%BLag5', 'team3PABLag5', 'team3PMBLag5', 'team3P%BLag5', 'teamFTABLag5', 'teamFTMBLag5',
                'teamFT%BLag5', 'teamORBBLag5', 'teamDRBBLag5', 'teamTRBBLag5', 'teamPTS1BLag5', 'teamPTS2BLag5',
                'teamPTS3BLag5', 'teamPTS4BLag5', 'teamPTS5BLag5', 'teamPTS6BLag5', 'teamPTS7BLag5', 'teamPTS8BLag5',
                'teamTREB%BLag5', 'teamASST%BLag5', 'teamTS%BLag5', 'teamE(effective)FG%BLag5', 'teamOREB%BLag5',
                'teamDREB%BLag5', 'teamTO%BLag5', 'teamSTL%BLag5', 'teamBLK%BLag5', 'teamBLKRBLag5', 'teamPPSBLag5',
                'teamFICBLag5', 'teamFIC40BLag5', 'teamOrtgBLag5', 'teamDrtgBLag5', 'teamEDiffBLag5', 'teamPlay%BLag5',
                'teamARBLag5', 'teamAST/TOBLag5', 'teamSTL/TOBLag5', 'pointspreadBLag4', 'teamLocBLag4', 'teamMinBLag4',
                'teamDayOffBLag4', 'teamPTSBLag4', 'teamASTBLag4', 'teamTOBLag4', 'teamSTLBLag4', 'teamBLKBLag4',
                'teamPFBLag4', 'teamFGABLag4', 'teamFGMBLag4', 'teamFG%BLag4', 'team2PABLag4', 'team2PMBLag4',
                'team2P%BLag4', 'team3PABLag4', 'team3PMBLag4', 'team3P%BLag4', 'teamFTABLag4', 'teamFTMBLag4',
                'teamFT%BLag4', 'teamORBBLag4', 'teamDRBBLag4', 'teamTRBBLag4', 'teamPTS1BLag4', 'teamPTS2BLag4',
                'teamPTS3BLag4', 'teamPTS4BLag4', 'teamPTS5BLag4', 'teamPTS6BLag4', 'teamPTS7BLag4', 'teamPTS8BLag4',
                'teamTREB%BLag4', 'teamASST%BLag4', 'teamTS%BLag4', 'teamE(effective)FG%BLag4', 'teamOREB%BLag4',
                'teamDREB%BLag4', 'teamTO%BLag4', 'teamSTL%BLag4', 'teamBLK%BLag4', 'teamBLKRBLag4', 'teamPPSBLag4',
                'teamFICBLag4', 'teamFIC40BLag4', 'teamOrtgBLag4', 'teamDrtgBLag4', 'teamEDiffBLag4', 'teamPlay%BLag4',
                'teamARBLag4', 'teamAST/TOBLag4', 'teamSTL/TOBLag4', 'pointspreadBLag3', 'teamLocBLag3', 'teamMinBLag3',
                'teamDayOffBLag3', 'teamPTSBLag3', 'teamASTBLag3', 'teamTOBLag3', 'teamSTLBLag3', 'teamBLKBLag3',
                'teamPFBLag3', 'teamFGABLag3', 'teamFGMBLag3', 'teamFG%BLag3', 'team2PABLag3', 'team2PMBLag3',
                'team2P%BLag3', 'team3PABLag3', 'team3PMBLag3', 'team3P%BLag3', 'teamFTABLag3', 'teamFTMBLag3',
                'teamFT%BLag3', 'teamORBBLag3', 'teamDRBBLag3', 'teamTRBBLag3', 'teamPTS1BLag3', 'teamPTS2BLag3',
                'teamPTS3BLag3', 'teamPTS4BLag3', 'teamPTS5BLag3', 'teamPTS6BLag3', 'teamPTS7BLag3', 'teamPTS8BLag3',
                'teamTREB%BLag3', 'teamASST%BLag3', 'teamTS%BLag3', 'teamE(effective)FG%BLag3', 'teamOREB%BLag3',
                'teamDREB%BLag3', 'teamTO%BLag3', 'teamSTL%BLag3', 'teamBLK%BLag3', 'teamBLKRBLag3', 'teamPPSBLag3',
                'teamFICBLag3', 'teamFIC40BLag3', 'teamOrtgBLag3', 'teamDrtgBLag3', 'teamEDiffBLag3', 'teamPlay%BLag3',
                'teamARBLag3', 'teamAST/TOBLag3', 'teamSTL/TOBLag3', 'pointspreadBLag2', 'teamLocBLag2', 'teamMinBLag2',
                'teamDayOffBLag2', 'teamPTSBLag2', 'teamASTBLag2', 'teamTOBLag2', 'teamSTLBLag2', 'teamBLKBLag2',
                'teamPFBLag2', 'teamFGABLag2', 'teamFGMBLag2', 'teamFG%BLag2', 'team2PABLag2', 'team2PMBLag2',
                'team2P%BLag2', 'team3PABLag2', 'team3PMBLag2', 'team3P%BLag2', 'teamFTABLag2', 'teamFTMBLag2',
                'teamFT%BLag2', 'teamORBBLag2', 'teamDRBBLag2', 'teamTRBBLag2', 'teamPTS1BLag2', 'teamPTS2BLag2',
                'teamPTS3BLag2', 'teamPTS4BLag2', 'teamPTS5BLag2', 'teamPTS6BLag2', 'teamPTS7BLag2', 'teamPTS8BLag2',
                'teamTREB%BLag2', 'teamASST%BLag2', 'teamTS%BLag2', 'teamE(effective)FG%BLag2', 'teamOREB%BLag2',
                'teamDREB%BLag2', 'teamTO%BLag2', 'teamSTL%BLag2', 'teamBLK%BLag2', 'teamBLKRBLag2', 'teamPPSBLag2',
                'teamFICBLag2', 'teamFIC40BLag2', 'teamOrtgBLag2', 'teamDrtgBLag2', 'teamEDiffBLag2', 'teamPlay%BLag2',
                'teamARBLag2', 'teamAST/TOBLag2', 'teamSTL/TOBLag2', 'pointspreadBLag1', 'teamLocBLag1', 'teamMinBLag1',
                'teamDayOffBLag1', 'teamPTSBLag1', 'teamASTBLag1', 'teamTOBLag1', 'teamSTLBLag1', 'teamBLKBLag1',
                'teamPFBLag1', 'teamFGABLag1', 'teamFGMBLag1', 'teamFG%BLag1', 'team2PABLag1', 'team2PMBLag1',
                'team2P%BLag1', 'team3PABLag1', 'team3PMBLag1', 'team3P%BLag1', 'teamFTABLag1', 'teamFTMBLag1',
                'teamFT%BLag1', 'teamORBBLag1', 'teamDRBBLag1', 'teamTRBBLag1', 'teamPTS1BLag1', 'teamPTS2BLag1',
                'teamPTS3BLag1', 'teamPTS4BLag1', 'teamPTS5BLag1', 'teamPTS6BLag1', 'teamPTS7BLag1', 'teamPTS8BLag1',
                'teamTREB%BLag1', 'teamASST%BLag1', 'teamTS%BLag1', 'teamE(effective)FG%BLag1', 'teamOREB%BLag1',
                'teamDREB%BLag1', 'teamTO%BLag1', 'teamSTL%BLag1', 'teamBLK%BLag1', 'teamBLKRBLag1', 'teamPPSBLag1',
                'teamFICBLag1', 'teamFIC40BLag1', 'teamOrtgBLag1', 'teamDrtgBLag1', 'teamEDiffBLag1', 'teamPlay%BLag1',
                'teamARBLag1', 'teamAST/TOBLag1', 'teamSTL/TOBLag1']
X = df.loc[:, feature_cols]
# print(X.shape)
X = X.to_numpy()
# print(X.shape)
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
print(X.shape)
Y = df['pointspread']
Y = Y.to_numpy()
print(Y.shape)

# before feature selection, model training with all features
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
# performance results
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('before feature selection, test accuracy:'+ str(acc))





# feature selection
from skfeature.function.structure import tree_fs
# specify the tree structure among features
idx = np.array([[-1, -1, 1],
                [1, 20, np.sqrt(20)],
                [21, 40, np.sqrt(20)],
                [41, 50, np.sqrt(10)],
                [51, 70, np.sqrt(20)],
                [71, 100, np.sqrt(30)],
                [1, 50, np.sqrt(50)],
                [51, 100, np.sqrt(50)]]).T
idx = idx.astype(int)
w, obj, value_gamma = tree_fs.tree_fs(X, Y, 0.01, idx, verbose=True) # here we set max_iter = 2321559
#print('weight vector:------------')
#print(w)
new_w = np.argsort(-w)
sorted_feature_cols = []
for index in new_w:
    print('feature column name:' + str(feature_cols[index]) + ', weight is:' + str(w[index]))
    sorted_feature_cols.append(feature_cols[index])

#print('objective function value during iterations:-----------')
#print(obj)
print('suitable step size during iterations:------------')
print(value_gamma)


XX = df.loc[:, sorted_feature_cols]
XX = XX.to_numpy()
min_max_scaler = preprocessing.MinMaxScaler()
XX = min_max_scaler.fit_transform(XX)
X_train, X_test, y_train, y_test = train_test_split(XX, Y, test_size=0.2, random_state=0)
num_fea = 100
selected_features_train = X_train[:, 0:num_fea]
selected_features_test = X_test[:, 0:num_fea]

# after feature selection, model training with selected features
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(selected_features_train, y_train)
y_predict = clf.predict(selected_features_test)
# performance results
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('after feature selection, test accuracy:'+ str(acc))


'''
from skfeature.utility import construct_W
kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
W = construct_W.construct_W(X, **kwargs_W)
from skfeature.function.similarity_based import lap_score
score = lap_score.lap_score(X, W=W)
print(score)
idx = lap_score.feature_ranking(score)
for index in idx:
    print(feature_cols[index],'---',score[index])
'''
