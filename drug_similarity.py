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

X_drug_DAVIS, X_target_DAVIS, y_DAVIS = load_process_KIBA('./data/', binary=False)

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

    fig, ax = plt.subplots(figsize=(5,5))
    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    ax.set_xlabel('Test') # testset
    ax.set_ylabel('Train') # trainset
    ax.set_title('Tanimoto Similarity Heat Map')
    plt.savefig('Tanimoto Similarity.png')


def distribution_generator(unique_drugs_train, unique_drugs_test):
    similarity_matrix_a = np.zeros((len(unique_drugs_train), len(unique_drugs_train)))
    similarity_matrix_b = np.zeros((len(unique_drugs_train), len(unique_drugs_test)))
    similarity_matrix_c = np.zeros((len(unique_drugs_test), len(unique_drugs_test)))

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
    plt.savefig('Tanimoto Similarity_Distribution_KIBA_Final.png')

unique_drugs_train_list = list(set(train_DAVIS['SMILES']))
unique_drugs_test_list = list(set(test_DAVIS['SMILES']))

unique_drugs_train = get_fingerprints(unique_drugs_train_list)
unique_drugs_test = get_fingerprints(unique_drugs_test_list)

distribution_generator(unique_drugs_train, unique_drugs_test)