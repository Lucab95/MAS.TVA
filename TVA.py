import csv
import pandas as pd
import string


df = pd.read_csv('voting_example2.csv', sep=";")


no_pref = df.columns.size-1
aDict = dict(zip(string.ascii_uppercase, [0] * no_pref))


for i in range(no_pref):
    for (index, j) in df.iloc[:, i+1].iteritems():
        aDict[j] += (no_pref - i - 1)
print(df)
print(aDict)