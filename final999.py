import numpy as np
from numpy import *
from nltk.corpus import wordnet as wd
import pandas as pd
from itertools import combinations
from nltk.corpus import stopwords
from scipy.sparse import csgraph
import pickle

with open('setsimlex999.data', 'rb') as f:
    change_dictionary = pickle.load(f)


n = len(change_dictionary)
#n = 100
S_ARelation = np.zeros((n,n))
m=0
k=0

for (wordi, wordj) in combinations(change_dictionary[0:n-1],2):
    i = change_dictionary.index(wordi)
    j = change_dictionary.index(wordj)
    for syn in wd.synsets(wordi):
        for l in syn.lemmas():
            if l.name() == wordj:
                S_ARelation[i][j] = 0.95
                S_ARelation[j][i] = 0.95
                m+=1
            if l.antonyms():
                if wordj == l.antonyms()[0].name():
                    S_ARelation[i][j] = 0.1
                    S_ARelation[j][i] = 0.1
                    k+=1

print(m,k)
#S_AFinal = csgraph.laplacian(S_ARelation, normed=True)
with open('Adj.data', 'rb') as f:
    Adj = pickle.load(f)

Final = S_ARelation + Adj
S_AFinal = csgraph.laplacian(Final, normed=True)

for i in range(0,n-1):
    Final[i][i] = 1.0


np.savetxt('Final2T_L-%d.txt' % n, S_AFinal, delimiter="\t")