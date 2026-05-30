import pandas as pd
import os
import numpy as np

path = 'data/Makoto2/Akuma3/'
filelist = os.listdir(path)

file_num = len(filelist)
min_posY = np.zeros(file_num)
min_posX = np.zeros(file_num)
max_posX = np.zeros(file_num)
max_posY = np.zeros(file_num)
max_health = np.zeros(file_num)
max_stunP1, max_stunP2 = np.zeros(file_num), np.zeros(file_num)
max_meterP1, max_meterP2 = np.zeros(file_num), np.zeros(file_num)
for i in range(file_num):
    raw_data = pd.read_csv(path + filelist[i])
    posY = raw_data['PosY']
    posX = raw_data['PosX']
    p1_data = raw_data.loc[raw_data['Player'] == 'P1']
    p2_data = raw_data.loc[raw_data['Player'] == 'P2']
    min_posY[i] = posY.min()
    min_posX[i] = posX.min()
    max_posY[i] = posY.max()
    max_posX[i] = posX.max()
    max_health[i] = raw_data['Health'].max()
    max_stunP1[i] = p1_data['Stun'].max()
    max_stunP2[i] = p2_data['Stun'].max()
    max_meterP1[i] = p1_data['Meter'].max()
    max_meterP2[i] = p2_data['Meter'].max()

print('min posy: ' + str(min_posY.min()))
print(f'max posy: {max_posY.max()}')
print(f'min posx: {min_posX.min()}')
print(f'max posx: {max_posX.max()}')
print(f'max health: {max_health.max()}')
print(f'max stun P1: {max_stunP1.max()}')
print(f'max stun P2: {max_stunP2.max()}')
print(f'max meter P1: {max_meterP1.max()}') 
print(f'max meter P2: {max_meterP2.max()}') 