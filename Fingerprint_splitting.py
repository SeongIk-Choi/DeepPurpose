import os 
aff = os.sched_getaffinity(0)

from rdkit import Chem
from DeepPurpose import DTI as models2
from DeepPurpose.utils import *
from DeepPurpose.dataset import *
import matplotlib.pyplot as plt
from ast import arg
from DeepPurpose.hpo_worker import BaseWorker
from DeepPurpose.simple_hyperband import HyperBand
import torch
os.sched_setaffinity(0,aff)
from rdkit import DataStructs
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.model_selection import train_test_split
from rdkit.Chem import rdMolDescriptors
import deepchem as dc
import numpy as np


X_drug_DAVIS, X_target_DAVIS, y_DAVIS = load_process_KIBA('./data/', binary=False)

def fingerprint_split(smiles_list):

    Xs = np.zeros(len(X_drug_DAVIS))
    # creation of a deepchem dataset with the smile codes in the ids field
    dataset = dc.data.DiskDataset.from_numpy(X=Xs,ids=X_drug_DAVIS)
    fingerprintsplitter = dc.splits.FingerprintSplitter()
    train_dataset, val_dataset, test_dataset = fingerprintsplitter.train_valid_test_split(dataset, frac_train=0.7, frac_val=0.1, frac_test=0.2)
    return train_dataset.ids, test_dataset.ids

def get_fingerprints(smiles_list):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    fingerprints = [rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols]
    return fingerprints

def distribution_generator(unique_drugs_train, unique_drugs_test):
    similarity_matrix_a = np.zeros((len(unique_drugs_train), len(unique_drugs_train)))
    similarity_matrix_b = np.zeros((len(unique_drugs_train), len(unique_drugs_test)))
    similarity_matrix_c = np.zeros((len(unique_drugs_test), len(unique_drugs_test)))

    # vector_a = [] 
    # vector_b = []

    # for i in range(len(unique_drugs_train)):
    #     vector_a.append(unique_drugs_train[i])

    # for j in range(len(unique_drugs_test)):
    #     vector_b.append(unique_drugs_test[j])

    # For train-train sets
    for i in range(len(unique_drugs_train)):
        for j in range(len(unique_drugs_train)):
            similarity_a = DataStructs.TanimotoSimilarity(unique_drugs_train[i], unique_drugs_train[j])
            similarity_matrix_a[i,j] = similarity_a

    # For train-test sets
    for i in range(len(unique_drugs_train)):
        for j in range(len(unique_drugs_test)):
            similarity_b = DataStructs.TanimotoSimilarity(unique_drugs_train[i], unique_drugs_test[j])
            similarity_matrix_b[i,j] = similarity_b

    # For test-test sets
    for i in range(len(unique_drugs_test)):
        for j in range(len(unique_drugs_test)):
            similarity_c = DataStructs.TanimotoSimilarity(unique_drugs_test[i], unique_drugs_test[j])
            similarity_matrix_c[i,j] = similarity_c

    # Assume similarity_matrix contains your similarity scores
    lower_triangular_a = [similarity_matrix_a[i][j] for i in range(len(unique_drugs_train)) for j in range(i)]
    #lower_triangular_a = [similarity_matrix_a[i,j] for i in range(len(vector_a)) for j in range(len(vector_a))]
    lower_triangular_b = [similarity_matrix_b[i,j] for i in range(len(unique_drugs_train)) for j in range(len(unique_drugs_test)) if j<i]
    lower_triangular_c = [similarity_matrix_c[i][j] for i in range(len(unique_drugs_test)) for j in range(i)]
    #lower_triangular_c = [similarity_matrix_c[i,j] for i in range(len(vector_b)) for j in range(len(vector_b))]


    # Plot the three distributions
    sns.kdeplot(np.array(lower_triangular_a), color='blue', label='train', shade=True)
    sns.kdeplot(np.array(lower_triangular_b), color='green', label='train-test', shade=True)
    sns.kdeplot(np.array(lower_triangular_c), color='red', label='test', shade=True)

    # Set the x-label, y-label, and title of the plot
    plt.xlabel('Similarity score')
    plt.ylabel('Density')
    plt.title('Smooth distribution of Tanimoto similarity scores')

    # Set the legend
    plt.legend()

    # Save the figure
    plt.savefig('Tanimoto Similarity_Distribution_fingerprint_KIBA_Final.png')

smiles_list = X_drug_DAVIS

train_smiles, test_smiles = fingerprint_split(smiles_list)

unique_drugs_train_list = list(set(train_smiles))
unique_drugs_test_list = list(set(test_smiles))

unique_drugs_train = get_fingerprints(unique_drugs_train_list)
unique_drugs_test = get_fingerprints(unique_drugs_test_list)

figure = distribution_generator(unique_drugs_train, unique_drugs_test)