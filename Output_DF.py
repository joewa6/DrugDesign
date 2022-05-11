#!/usr/bin/env python
# coding: utf-8

# In[4]:

import os
import sys
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
import rdkit

# import RDKit ----------------------------------------------------------------
from rdkit.Chem import RDKFingerprint
from rdkit.Chem import rdMolDescriptors

# import numpy for data type conversion ---------------------------------------
print('rdkit: {}'.format(rdkit.__version__))
#from rdkit.Chem import rdMolDraw2D
# Load libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
# In[5]:

dataset = read_csv("dataset.csv" ,sep=';', low_memory=False)
print(dataset.columns.tolist())

df = dataset
df= df[['pChEMBL Value','Molecule ChEMBL ID','Molecular Weight', 'AlogP','Smiles','Standard Value' ]].copy()
####____________________________________________________________________________
###### REDUCE TOTAL DATA TO TEST dataset

df= df.head(3)
df = df.sort_values('pChEMBL Value', ascending=False)
###########################
df.to_csv("DF_OUT.txt", sep='\t')
print(df)

'''
mol2_files = []
for n,i in enumerate(df['Smiles']):
    print(i)
    print(n)

    os.system("obabel -:'%s' -O lx_%s.mol2 --gen3d" %(i, n))
    mol2_files.append(str("lx_%s.mol2" %(n)))

with open ("ligands", "w") as f:
    for i in mol2_files:
        f.write(i + "\n")

os.system("bash do_dock.sh")
#obabel -:"O=C(NCC(c1ccco1)N1CCN(c2ccc(F)cc2)CC1)c1ccco1" -O lx.mol2 --gen3d
'''






'''
Fingerprints = []

for i in df['Smiles']:
    smile = str(i)
    mol = Chem.MolFromSmiles(smile)
    fingerprint_rdk = RDKFingerprint(mol)
    fingerprint_rdk_np = np.array(fingerprint_rdk)
    Fingerprints.append(fingerprint_rdk_np[0])

df["Fingerprints"] = Fingerprints

'''
