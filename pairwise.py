import pandas as pd
import numpy as np
import itertools
import csv
from scipy.stats import spearmanr

def list_of_combinations(choose_n, data):
    combinations = []
    for comb in itertools.combinations(data, choose_n):
        combinations.append(comb)
    return combinations

def pairwise_interactions_product(combinations, data):
    result=[]
    names=[]
    for comb in combinations:
        # get values of the choosen two features
        x = data.loc[:, comb[0]]
        y = data.loc[:, comb[1]]
        # calculate correlation of the features in healthy cell lines
        x1=x[0:29,]
        y1=y[0:29,]
        rho, pval = spearmanr(x1,y1)
        # if the abs of correlation between the features is more than 0.5 it is worth consideration
        if abs(rho) > 0.9:
            result.append(np.asarray(x*y))
            names.append(comb)
    df = pd.DataFrame(data=result).T
    df.columns = names
    return df


cer_cancer=pd.read_csv("cervical.csv", index_col=False)
cer_cancer=cer_cancer.T
cer_cancer.columns = cer_cancer.iloc[0]
cer_cancer=cer_cancer.reindex(cer_cancer.index.drop("ID"))

zeros_values=np.zeros(29) #First 29 sampples are healthy
ones_values=np.ones(29)  #Next 29 samples are cancerous
class_values=np.append(zeros_values,ones_values)
cer_cancer_data= cer_cancer
cer_cancer_data=cer_cancer_data.assign(label=class_values)

cer_cancer_data=cer_cancer_data.loc[:,'let-7a':'miR-1228']
healthy_samples = cer_cancer_data.loc['N1':'N29',:]
index_feature_to_delete=np.where(healthy_samples.std()==0)
cer_cancer_data=cer_cancer_data.drop(cer_cancer_data.columns[index_feature_to_delete[0]], axis=1)

pairwise_interactions = list_of_combinations(2, cer_cancer_data)
standarize = lambda x: (x-x.mean()) / x.std()
cer_cancer_data=cer_cancer_data.pipe(standarize)

df_interactions=pairwise_interactions_product(pairwise_interactions, cer_cancer_data)
df_interactions.to_csv("interactions_0.csv",header=True, index=False)

