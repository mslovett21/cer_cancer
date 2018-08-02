"""
This script runs CCM on three example datasets. 
These datasets correspond to the synthetic datasets in the paper.

Type of the datasets are binary classification, 
categorical classification, and regression respectively.
"""
from __future__ import print_function
import numpy as np
from make_synthetic_data import generate_data
import sys
sys.path.append('core')
import ccm 
import csv


def readcsv(filename):
    ifile = open(filename, "r")
    reader = csv.reader(ifile, delimiter=",")
    rownum = 0
    a = []
    for row in reader:
        a.append (row)
        rownum += 1   
    ifile.close()
    return a

print('Running CCM on the cervical cancer dataset...')
X =readcsv('examples/interactions4.csv')
pair_names=X[0]
del X[0]
X= np.asarray(X)
X=X.astype(np.float)

zeros_values=np.zeros(29) #First 29 sampples are healthy
ones_values=np.ones(29)  #Next 29 samples are cancerous
Y=np.append(zeros_values,ones_values)


epsilon = 0.001; num_features = 50; type_Y = 'binary'
rank = ccm.ccm(X, Y, num_features, type_Y, 
	epsilon, iterations = 100, verbose = False)
selected_feats = np.argsort(rank)[:50]
print('The features selected by CCM on the cervical cancer dataset are features {}'.format(selected_feats))
for feature in selected_feats:
	print(pair_names[feature])

