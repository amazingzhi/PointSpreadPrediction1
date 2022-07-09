import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#upload dataset
df = pd.read_csv('../data/games and player from 2004 to 2020/game_prediction_data/0420_with_back82_back41_back3_back2_SumPMs.csv')
dataall = df.iloc[:,8:]
rest = df.iloc[:,:8]
HvsALag1 = df.iloc[:,8:50]
HvsALag2 = df.iloc[:,50:92]
HLag1 = df.iloc[:,92:146]
ALag1 = df.iloc[:,146:200]
HLag2 = df.iloc[:,200:254]
ALag2 = df.iloc[:,254:308]
HLag3 = df.iloc[:,308:362]
ALag3 = df.iloc[:,362:416]
HAvg82 = df.iloc[:,416:445]
AAvg82 = df.iloc[:,445:474]
HAvg41 = df.iloc[:,474:515]
AAvg41 = df.iloc[:,515:]
data = [HvsALag1,HvsALag2,HLag1,ALag1,HLag2,ALag2,HLag3,ALag3,HAvg82,AAvg82,HAvg41,AAvg41]
sc = StandardScaler()

# for i in data:
#     sc.fit(i)
#     std = sc.transform(i)
#     pca = PCA().fit(std)
#     plt.plot(np.cumsum(pca.explained_variance_ratio_), label = f'{i.columns[0]}')
#     plt.legend()
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
final_df = []

columns = []
for j in range(1,16):
    columns.append(f'AvsBLagOne{j}')
std = sc.fit_transform(HvsALag1)
pca = PCA(n_components=15)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([rest,tempDF],axis=1)

columns = []
for j in range(1,16):
    columns.append(f'AvsBLagTwo{j}')
std = sc.fit_transform(HvsALag2)
pca = PCA(n_components=15)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

columns = []
for j in range(1,19):
    columns.append(f'HLagOne{j}')
std = sc.fit_transform(HLag1)
pca = PCA(n_components=18)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

columns = []
for j in range(1,19):
    columns.append(f'ALagOne{j}')
std = sc.fit_transform(ALag1)
pca = PCA(n_components=18)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

columns = []
for j in range(1,19):
    columns.append(f'HLagTwo{j}')
std = sc.fit_transform(HLag2)
pca = PCA(n_components=18)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

columns = []
for j in range(1,19):
    columns.append(f'ALagTwo{j}')
std = sc.fit_transform(ALag2)
pca = PCA(n_components=18)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

columns = []
for j in range(1,19):
    columns.append(f'HLagThree{j}')
std = sc.fit_transform(HLag3)
pca = PCA(n_components=18)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

columns = []
for j in range(1,19):
    columns.append(f'ALagThree{j}')
std = sc.fit_transform(ALag3)
pca = PCA(n_components=18)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

columns = []
for j in range(1,13):
    columns.append(f'HAvg82{j}')
std = sc.fit_transform(HAvg82)
pca = PCA(n_components=12)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

columns = []
for j in range(1,13):
    columns.append(f'AAvg82{j}')
std = sc.fit_transform(AAvg82)
pca = PCA(n_components=12)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

columns = []
for j in range(1,16):
    columns.append(f'HAvg41{j}')
std = sc.fit_transform(HAvg41)
pca = PCA(n_components=15)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

columns = []
for j in range(1,16):
    columns.append(f'AAvg41{j}')
std = sc.fit_transform(AAvg41)
pca = PCA(n_components=15)
Pcaed = pca.fit_transform(std)
tempDF = pd.DataFrame(data=Pcaed, columns=columns)
final_df = pd.concat([final_df,tempDF],axis=1)

resultCSVPath = r'C:\Users\jliu471\Desktop\result.csv'
final_df.to_csv(resultCSVPath,index = False,na_rep = 0)