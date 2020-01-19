import pandas as pd
import numpy as np
import re

data = [1, 2, 3, 4]
###################################################
# Create a series from a list
series = pd.Series(data)
print(series)
# 0    1
# 1    2
# 2    3
# 3    4
# dtype: int64

###################################################
# Change the indexing
series = pd.Series(data, index=['a', 'b', 'c', 'd'])
print(series)
# a    1
# b    2
# c    3
# d    4
# dtype: int64

###################################################
# Create a dataframe from a list
df = pd.DataFrame(data)
print(df)
#    0
# 0  1
# 1  2
# 2  3
# 3  4

###################################################
# Create a dataframe from a dictionary
data2 = {'echipe': ['Manchester United',
                    'Real Madrid', 'AC Milan'], 'trofee': [21, 15, 6]}
df = pd.DataFrame(data2)
print(df)
#               echipe  trofee
# 0  Manchester United      21
# 1        Real Madrid      15
# 2           AC Milan       6

###################################################
# Create a dataframe from a series
series = pd.Series([6, 12], index=['a', 'b'])
df = pd.DataFrame(series)
print(df)
#     0
# a   6
# b  12

###################################################
# Create a dataframe from a numpy array
na = np.array([[50000, 60000], ['John', 'James']])
df = pd.DataFrame({'name': na[1], 'wage': na[0]})
print(df)
#     name   wage
# 0   John  50000
# 1  James  60000

###################################################
# Merge/Join and concatenate
player = ['Ronaldo', 'Pogba', 'De Bruyne']
trophies = [13, 12, 6]
team = ['Juve', 'ManU', 'ManC']
df1 = pd.DataFrame({'Player': player, 'Trophies': trophies, 'Team': team})
print(df1)
#       Player  Trophies  Team
# 0    Ronaldo        13  Juve
# 1      Pogba        12  ManU
# 2  De Bruyne         6  ManC

player = ['Ronaldo', 'Messi', 'Hazard']
foot = ['Both', 'Left', 'Right']
team = ['Juve', 'Barc', 'Real']
df2 = pd.DataFrame({'Player': player, 'Foot': foot, 'Team': team})
print(df2)
#     Player   Foot  Team
# 0  Ronaldo   Both  Juve
# 1    Messi   Left  Barc
# 2   Hazard  Right  Real

# Inner Merge
print(df1.merge(df2, on='Player', how='inner'))
#     Player  Trophies Team_x  Foot Team_y
# 0  Ronaldo        13   Juve  Both   Juve

# Left/Right Merge
print(df1.merge(df2, on='Player', how='left'))
#       Player  Trophies Team_x  Foot Team_y
# 0    Ronaldo        13   Juve  Both   Juve
# 1      Pogba        12   ManU   NaN    NaN
# 2  De Bruyne         6   ManC   NaN    NaN

# Outer Merge
print(df1.merge(df2, on='Player', how='outer'))
#       Player  Trophies Team_x   Foot Team_y
# 0    Ronaldo      13.0   Juve   Both   Juve
# 1      Pogba      12.0   ManU    NaN    NaN
# 2  De Bruyne       6.0   ManC    NaN    NaN
# 3      Messi       NaN    NaN   Left   Barc
# 4     Hazard       NaN    NaN  Right   Real

player = ['Ronaldo', 'Pogba', 'De Bruyne']
trophies = [13, 12, 6]
team = ['Juve', 'ManU', 'ManC']
df1 = pd.DataFrame({'Player': player, 'Trophies': trophies,
                    'Team': team}, index=['L1', 'L2', 'L3'])
player = ['Ronaldo', 'Messi', 'Hazard']
foot = ['Both', 'Left', 'Right']
team = ['Juve', 'Barc', 'Real']
df2 = pd.DataFrame({'Players': player, 'Foot': foot,
                    'Teams': team}, index=['L2', 'L3', 'L4'])
# Outer join
print(df1.join(df2, how='outer'))
#        Player  Trophies  Team  Players   Foot Teams
# L1    Ronaldo      13.0  Juve      NaN    NaN   NaN
# L2      Pogba      12.0  ManU  Ronaldo   Both  Juve
# L3  De Bruyne       6.0  ManC    Messi   Left  Barc
# L4        NaN       NaN   NaN   Hazard  Right  Real

# Concatenate
print(pd.concat([df1, df2]))
#      Foot     Player  Players  Team Teams  Trophies
# L1    NaN    Ronaldo      NaN  Juve   NaN      13.0
# L2    NaN      Pogba      NaN  ManU   NaN      12.0
# L3    NaN  De Bruyne      NaN  ManC   NaN       6.0
# L2   Both        NaN  Ronaldo   NaN  Juve       NaN
# L3   Left        NaN    Messi   NaN  Barc       NaN
# L4  Right        NaN   Hazard   NaN  Real       NaN

###################################################
# Reading a .csv file
file = pd.read_csv('football_players.txt')
print(file.head(3))  # head(5)/tail(10)
#    index      player               team  rating  age      wage  Unnamed: 6
# 0      1       Burki  Manchester United      88   19  49900000         NaN
# 1      2   McTominay            Arsenal      91   24  19100000         NaN
# 2      3  Di Lorenzo           Juventus      95   18  49400000         NaN

print(file.shape)
# (176, 7)

print(file.mean())  # max()/min()/std()/median()/describe()
# index         8.850000e+01
# rating        8.901136e+01
# age           2.579545e+01
# wage          2.678693e+07
# Unnamed: 6             NaN
# dtype: float64

###################################################
# Rename one column
file = file.rename(columns={'player': 'Player'})
#      index      Player               team  rating  age      wage  Unnamed: 6
# 0        1       Burki  Manchester United      88   19  49900000         NaN
# 1        2   McTominay            Arsenal      91   24  19100000         NaN
# 2        3  Di Lorenzo           Juventus      95   18  49400000         NaN
# 3        4    Torreira          Barcelona      90   19   5700000         NaN
# 4        5       Ayoze            Chelsea      80   28  27800000         NaN
# ..     ...         ...                ...     ...  ...       ...         ...

###################################################
# Fill the NA cells with one value
file.rating = file.rating.fillna(file.rating.mean())

###################################################
# Delete one column
file = file.drop(columns=['Unnamed: 6'])

###################################################
# Make correlations
df = df[['rating', 'wage', 'age']].corr()
#           rating      wage       age
# rating  1.000000 -0.006147  0.004977
# wage   -0.006147  1.000000 -0.027510
# age     0.004977 -0.027510  1.000000

###################################################
# Convert the type of one column
file.rating = file.rating.astype(float)
print(file.info(null_counts=True))
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 176 entries, 0 to 175
# Data columns (total 7 columns):
# index         176 non-null int64
# player        176 non-null object
# team          176 non-null object
# rating        176 non-null float64
# age           176 non-null int64
# wage          176 non-null int64
# Unnamed: 6    0 non-null float64
# dtypes: float64(2), int64(3), object(2)
# memory usage: 9.8+ KB

###################################################
# View a sigle column
print(file.iloc[:, 1])
print(file['player'][:])
# 0           Burki
# 1       McTominay
# 2      Di Lorenzo
# 3        Torreira
# 4           Ayoze
#           ...

###################################################
# View first 5 records of a single column
print(file.iloc[0:5, 1])
# 0         Burki
# 1     McTominay
# 2    Di Lorenzo
# 3      Torreira
# 4         Ayoze
# Name: player, dtype: object

###################################################
# View all rows/columns
print(file.iloc[:, :])

###################################################
# Iterate through rows
for index, row in file.iterrows():
    print(index, row['player'], row['team'])
# 0 Burki Manchester United
# 1 McTominay Arsenal
# 2 Di Lorenzo Juventus
# 3 Torreira Barcelona

###################################################
# Filter by one column name
print(file.loc[file['team'] == 'Manchester United'])

###################################################
# View by inserting the column name
print(file.loc[:, ['player', 'team']])
print(file[['player', 'team']])
#          player               team
# 0         Burki  Manchester United
# 1     McTominay            Arsenal
# 2    Di Lorenzo           Juventus
# 3      Torreira          Barcelona
# 4         Ayoze            Chelsea
# ..          ...                ...
# 171        Leno          Barcelona
# 172     Sanchez           Dortmund
# 173    Cuadrado                PSG
# 174       Allan            Chelsea
# 175   Alcantara  Manchester United
#
# [176 rows x 2 columns]

print(file.loc[:, 'player':'rating'])
#          player               team  rating
# 0         Burki  Manchester United      88
# 1     McTominay            Arsenal      91
# 2    Di Lorenzo           Juventus      95
# 3      Torreira          Barcelona      90
# 4         Ayoze            Chelsea      80
# ..          ...                ...     ...
# 171        Leno          Barcelona      83
# 172     Sanchez           Dortmund      82
# 173    Cuadrado                PSG      80
# 174       Allan            Chelsea      88
# 175   Alcantara  Manchester United      82
#
# [176 rows x 3 columns]

###################################################
# Creating a new column
file['from_earth'] = 'yes'
print(file.iloc[:3, :])
#    index      player               team  ...      wage  Unnamed: 6  from_earth
# 0      1       Burki  Manchester United  ...  49900000         NaN         yes
# 1      2   McTominay            Arsenal  ...  19100000         NaN         yes
# 2      3  Di Lorenzo           Juventus  ...  49400000         NaN         yes
#
# [3 rows x 8 columns]

###################################################
# Create a column with lambda function
f = lambda x: x*2
file['from_earth'] = 1
file['from_earth'] = file['from_earth'].apply(f)
print(file.iloc[:3, :])
#    index      player               team  ...      wage  Unnamed: 6  from_earth
# 0      1       Burki  Manchester United  ...  49900000         NaN           2
# 1      2   McTominay            Arsenal  ...  19100000         NaN           2
# 2      3  Di Lorenzo           Juventus  ...  49400000         NaN           2
#
# [3 rows x 8 columns]

file['Info'] = file['player'] + ' player of ' + file['team']
print(file.iloc[:3, :])
#      index      player  ... Unnamed: 6                                   Info
# 0        1       Burki  ...        NaN      Burki player of Manchester United
# 1        2   McTominay  ...        NaN            McTominay player of Arsenal
# 2        3  Di Lorenzo  ...        NaN          Di Lorenzo player of Juventus

###################################################
# Sort a column
print(file.sort_values(by=['rating', 'age'], ascending=(0, 1)).iloc[:, :])
#      index      player               team  rating  age      wage  Unnamed: 6
# 67      68  Handanovic        Real Madrid      99   22  14000000         NaN
# 41      42   Guendouzi                PSG      99   26  13300000         NaN
# 141    142    Chilwell             Napoli      99   29  30800000         NaN
# 160    161     Manolas          Barcelona      99   29  36700000         NaN
# 28      29     Rakitic     Bayern Munchen      99   32  40400000         NaN

###################################################
# Print filtered data
filter1 = file['rating'] > 98
filtered = file[filter1]
print(filtered)
#      index      player            team  rating  age      wage  Unnamed: 6
# 28      29     Rakitic  Bayern Munchen      99   32  40400000         NaN
# 41      42   Guendouzi             PSG      99   26  13300000         NaN
# 67      68  Handanovic     Real Madrid      99   22  14000000         NaN
# 141    142    Chilwell          Napoli      99   29  30800000         NaN
# 160    161     Manolas       Barcelona      99   29  36700000         NaN

# Multiple filtering
filter2 = (file['rating'] > 98) & (file['age'] > 28)
filtered = file[filter2]
print(filtered)
#      index    player            team  rating  age      wage  Unnamed: 6
# 28      29   Rakitic  Bayern Munchen      99   32  40400000         NaN
# 141    142  Chilwell          Napoli      99   29  30800000         NaN
# 160    161   Manolas       Barcelona      99   29  36700000         NaN

print(file.loc[(file['team'] == 'Manchester United')
               & (file['age'] < 25) & (file['rating'] > 90)])
#      index    player               team  rating  age      wage  Unnamed: 6
# 9       10    Morata  Manchester United      95   21  20100000         NaN
# 27      28  Sokratis  Manchester United      91   24  40200000         NaN
# 52      53    Aurier  Manchester United      96   20  21000000         NaN
# 138    139   Insigne  Manchester United      95   20  30300000         NaN

###################################################
# Filter using REGEX
print(file.loc[(file['player'].str.contains('Ronaldo'))])
print(file.loc[~file['player'].str.contains('Pogba')])
print(file.loc[file['team'].str.contains(
    'Manchester|REAL', flags=re.I, regex=True)].sort_values(by=['team'], ascending=True))
#      index      player               team  rating  age      wage  Unnamed: 6
# 101    102   Tielemans    Manchester City      81   21  20000000         NaN
# 80      81   Robertson    Manchester City      97   18  44400000         NaN
# 77      78       Alaba    Manchester City      91   25  16900000         NaN
# 66      67      Muller    Manchester City      80   28  12400000         NaN
# 50      51       Sensi    Manchester City      86   27  41200000         NaN
# 0        1       Burki  Manchester United      88   19  49900000         NaN
# 138    139     Insigne  Manchester United      95   20  30300000         NaN
# 109    110      Buffon  Manchester United      84   19  34100000         NaN

###################################################
# Save a new csv
file.to_csv('modified.csv', index=False)  # csv
file.to_csv('modified.txt', index=False, sep='\t')  # txt

###################################################
# Reset index
file = file.reset_index(drop=True)
file.reset_index(drop=True, inplace=True)

###################################################
# Change value of a column
file.loc[file['team'] == 'Napoli', 'team'] = '1Steaua'
print(file.sort_values(by='team', ascending=True))
#      index    player       team  rating  age      wage  Unnamed: 6
# 113    114    Bernat    1Steaua      86   31  25100000         NaN
# 73      74     Pique    1Steaua      89   20   5400000         NaN
# 163    164    Lloris    1Steaua      81   25  13900000         NaN
# 141    142  Chilwell    1Steaua      99   29  30800000         NaN
# 75      76    Dybala    1Steaua      81   26  27800000         NaN
# ..     ...       ...        ...     ...  ...       ...         ...

###################################################
# Group rows
print(file.groupby(['team']).mean().sort_values(
    'rating', ascending=False))  # .mean/.sum/.count
#                         index     rating        age          wage  Unnamed: 6
# team
# Juventus            84.750000  91.500000  24.375000  2.265000e+07         NaN
# Arsenal             99.000000  91.000000  24.333333  2.226667e+07         NaN
# Atletico Madrid     79.923077  90.307692  27.461538  2.093077e+07         NaN
# Manchester United   64.166667  89.583333  24.083333  3.104167e+07         NaN
# Real Madrid         35.555556  89.555556  25.777778  2.562222e+07         NaN
# Chelsea             94.684211  89.421053  27.473684  2.776842e+07         NaN
# Bayern Munchen      98.900000  89.400000  24.800000  2.773000e+07         NaN
# PSG                 76.888889  89.333333  25.222222  2.243333e+07         NaN
# Liverpool          102.230769  89.076923  25.769231  2.555385e+07         NaN
# Inter Milan         97.625000  88.750000  28.375000  2.797500e+07         NaN
# Tottenham           98.000000  88.750000  27.500000  2.352500e+07         NaN
# Barcelona          104.714286  88.214286  26.500000  2.805000e+07         NaN
# Dortmund           105.500000  88.000000  23.722222  2.971111e+07         NaN
# Manchester City     75.800000  87.000000  23.800000  2.698000e+07         NaN
# Napoli              99.090909  86.818182  27.454545  3.115455e+07         NaN
# Leicester           53.285714  86.142857  25.285714  3.354286e+07         NaN

file['Count'] = 1
print(file.groupby(['team', 'rating']).count()['Count'])
# team       rating
# Arsenal    83        1
#            85        1
#            89        1
#            90        1
#            91        2
#                     ..
# Tottenham  88        1
#            90        1
#            91        1
#            92        1
#            97        1
# Name: Count, Length: 144, dtype: int64

###################################################
# Read in chunks
for file in pd.read_csv('football_players.txt', chunksize=2):
    print(file)
    print('*' * 70)
#    index     player               team  rating  age      wage  Unnamed: 6
# 0      1      Burki  Manchester United      88   19  49900000         NaN
# 1      2  McTominay            Arsenal      91   24  19100000         NaN
# **********************************************************************
#    index      player       team  rating  age      wage  Unnamed: 6
# 2      3  Di Lorenzo   Juventus      95   18  49400000         NaN
# 3      4    Torreira  Barcelona      90   19   5700000         NaN
# **********************************************************************
