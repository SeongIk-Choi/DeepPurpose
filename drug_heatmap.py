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
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from rdkit import DataStructs
import seaborn as sns
from rdkit.Chem import rdMolDescriptors

# Generate train, val, test dataset
drug_encoding = 'CNN'
target_encoding = 'CNN'

X_drug_DAVIS, X_target_DAVIS, y_DAVIS = load_process_DAVIS('./data/', binary=False)

train_DAVIS, val_DAVIS, test_DAVIS = data_process(X_drug_DAVIS, X_target_DAVIS, y_DAVIS,
                                drug_encoding, target_encoding,
                                split_method='cold_drug',frac=[0.7,0.1,0.2])

def get_fingerprints(smiles_list):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    fingerprints = [rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols]
    return fingerprints

def heatmap_generator(unique_drugs_train, unique_drugs_test):

    # create empty similarity matrix with shape (n_drugs_DAVIS, n_drugs_KIBA)
    similarity_matrix = np.zeros((len(unique_drugs_train), len(unique_drugs_test)))

    for i in range(len(unique_drugs_train)):
        for j in range(len(unique_drugs_test)):
            similarity = DataStructs.TanimotoSimilarity(unique_drugs_train[i], unique_drugs_test[j])
            similarity_matrix[i,j] = similarity

    return similarity_matrix 


unique_drugs_train_list = list(set(train_DAVIS['SMILES']))
unique_drugs_test_list = list(set(test_DAVIS['SMILES']))

unique_drugs_train = get_fingerprints(unique_drugs_train_list)
unique_drugs_test = get_fingerprints(unique_drugs_test_list)

# Create the similarity matrices
matrix_train_train = heatmap_generator(unique_drugs_train, unique_drugs_train)
matrix_train_test = heatmap_generator(unique_drugs_train, unique_drugs_test)
matrix_test_test = heatmap_generator(unique_drugs_test, unique_drugs_test)

# Create empty lines with zeros
blank_line_vertical = np.zeros((matrix_train_train.shape[0], 1))
blank_line_horizontal = np.zeros((1, matrix_train_test.shape[1]))

# Merge the matrices to form the combined heatmap matrix with blank lines
merged_matrix = np.block([
    [matrix_train_train, blank_line_vertical, matrix_train_test],
    [blank_line_horizontal, np.zeros((1, 1)), blank_line_vertical.T],
    [matrix_train_test.T, blank_line_horizontal.T, matrix_test_test]
])

# Plot the merged heatmap
fig, ax = plt.subplots(figsize=(5, 5))
plt.imshow(merged_matrix, cmap='hot', interpolation='nearest')
plt.colorbar()
ax.set_title("Tanimoto Similarity Heatmap")
plt.savefig("Merged_Tanimoto_Similarity.png")


