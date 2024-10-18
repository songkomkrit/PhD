#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import os
import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree

dim = 184 # data dimension
numsel = 10 # total selection
clf_depth = 5 # tree_depth

inputdir = f"input/dim-{dim}"
datfile = f"seltrain20num{dim}each20.csv"
datinfopattern = r'(selproc20).*(info.csv)'

outpostfix = f"dim-{dim}-sel-{numsel}-depth-{clf_depth}"

datinfo = None
for file in os.listdir(inputdir):
    if re.match(datinfopattern, file):
        datinfo = file
        break

dfcati = pd.read_csv(f"{inputdir}/{datinfo}")
feat_cat = list(dfcati[dfcati['type'] == 'Categorical']['variable'])

df = pd.read_csv(f"{inputdir}/{datfile}")

for v in feat_cat:
    df[v] = df[v].astype('category')

df.head()

one_hot_data = pd.get_dummies(df[feat_cat], drop_first=True)
X = df.iloc[:,0:-(len(feat_cat)+1)].join(one_hot_data)
y = df['class']

clf = DecisionTreeClassifier(max_depth=clf_depth, random_state=0)
clf.fit(X, y)

score = clf.score(X, y)
y_pred = clf.predict(X)
err_ind = (y_pred != y.to_numpy().flatten()).astype(int)
error = np.count_nonzero(err_ind)
accuracy = (1-error/len(y_pred))*100

df_pred = pd.DataFrame({'i': range(1,101)})
df_pred['y_true'] = df['class']
df_pred['y_pred'] = y_pred
df_pred['e'] = err_ind
df_pred.to_csv(f"output/pred-1hot-{outpostfix}.csv", index=False)

sumheader = ['error', 'accuracy', 'score']
summary = [{'error': error, 'accuracy': accuracy, 'score': score}]
with open(f"output/summary-1hot-{outpostfix}.csv", 'w') as file:
    writer = csv.DictWriter(file, fieldnames=sumheader)
    writer.writeheader()
    for row in summary:
        writer.writerow(row) 

print(f"Error = {error} | Accuracy = {accuracy} | Score = {score}")

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4,4), dpi=600)
tree.plot_tree(clf);
fig.savefig(f"images/plottreedefault-1hot-{outpostfix}.png")
