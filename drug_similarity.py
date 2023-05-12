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
# Generate train, val, test dataset
drug_encoding = 'CNN'
target_encoding = 'CNN'

X_drug_DAVIS, X_target_DAVIS, y_DAVIS = load_process_KIBA('./data/', binary=False)

train_DAVIS, val_DAVIS, test_DAVIS = data_process(X_drug_DAVIS, X_target_DAVIS, y_DAVIS,
                                drug_encoding, target_encoding,
                                split_method='cold_drug',frac=[0.7,0.1,0.2])

def smiles2morgan(s, radius = 2, nBits = 1024):
    mol = Chem.MolFromSmiles(s)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    arr = np.zeros((1,))
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def target2aac(s):
	try:
		features = CalculateAADipeptideComposition(s)
	except:
		print('AAC fingerprint not working for smiles: ' + s + ' convert to 0 vectors')
		features = np.zeros((8420, ))
	return np.array(features)

def tanimoto_similarity(a, b):
    """
    Calculate the Tanimoto similarity coefficient between two binary feature sets a and b.
    """
    # Calculate the intersection and union of the two sets

    intersection = len(set(a) & (set(b)))
    union = len(set(a) | (set(b)))
    # Calculate the Tanimoto coefficient
    if union == 0:
        return 0.0
    else:
        return float(intersection) / float(union)

unique_drugs_train = list(set(train_DAVIS['SMILES']))
unique_drugs_test = list(set(test_DAVIS['SMILES']))

def heatmap_generator(unique_drugs_train, unique_drugs_test):

    vector_a = [] 
    vector_b = []

    for i in range(len(unique_drugs_train)):
        vector_a.append(unique_drugs_train[i])

    for j in range(len(unique_drugs_test)):
        vector_b.append(unique_drugs_test[j])

    # create empty similarity matrix with shape (n_drugs_DAVIS, n_drugs_KIBA)
    similarity_matrix = np.zeros((len(unique_drugs_train), len(unique_drugs_test)))

    for i in range(len(vector_a)):
        for j in range(len(vector_b)):
            similarity = tanimoto_similarity(vector_a[i], vector_b[j])
            similarity_matrix[i,j] = similarity

    fig, ax = plt.subplots(figsize=(5,5))
    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    ax.set_xlabel('Test') # testset
    ax.set_ylabel('Train') # trainset
    ax.set_title('Tanimoto Similarity Heat Map')
    plt.savefig('Tanimoto Similarity_Train_Train_B.png')


def distribution_generator(unique_drugs_train, unique_drugs_train):
    similarity_matrix_a = np.zeros((len(unique_drugs_train), len(unique_drugs_train)))
    similarity_matrix_b = np.zeros((len(unique_drugs_train), len(unique_drugs_test)))
    similarity_matrix_c = np.zeros((len(unique_drugs_test), len(unique_drugs_test)))

    vector_a = [] 
    vector_b = []

    for i in range(len(unique_drugs_train)):
        vector_a.append(unique_drugs_train[i])

    for j in range(len(unique_drugs_test)):
        vector_b.append(unique_drugs_test[j])
    
    # For train-train sets
    for i in range(len(vector_a)):
        for j in range(len(vector_a)):
            similarity_a = tanimoto_similarity(vector_a[i], vector_a[j])
            similarity_matrix_a[i,j] = similarity_a

    # For train-test sets
    for i in range(len(vector_a)):
        for j in range(len(vector_b)):
            similarity_b = tanimoto_similarity(vector_a[i], vector_b[j])
            similarity_matrix_b[i,j] = similarity_b

    # For test-test sets
    for i in range(len(vector_b)):
        for j in range(len(vector_b)):
            similarity_c = tanimoto_similarity(vector_b[i], vector_b[j])
            similarity_matrix_c[i,j] = similarity_c

    # # calculate similarity between all pairs of drugs
    # for i in range(len(unique_drugs_DAVIS)):
    #     for j in range(len(unique_drugs_KIBA)):
    #         similarity = cosine_similarity(smiles2morgan(unique_drugs_DAVIS[i]).reshape(1,-1), smiles2morgan(unique_drugs_KIBA[j]).reshape(1,-1))
    #         similarity_matrix[i,j] = similarity

    # Assume similarity_matrix contains your similarity scores
    lower_triangular_a = [similarity_matrix_a[i,j] for i in range(len(vector_a)) for j in range(len(vector_a))]
    lower_triangular_b = [similarity_matrix_b[i,j] for i in range(len(vector_a)) for j in range(len(vector_b)) if j<i]
    lower_triangular_c = [similarity_matrix_c[i,j] for i in range(len(vector_b)) for j in range(len(vector_b))]

    # Plot the three distributions
    sns.kdeplot(np.array(lower_triangular_a), color='blue', label='train-train', shade=True)
    sns.kdeplot(np.array(lower_triangular_b), color='green', label='train-test', shade=True)
    sns.kdeplot(np.array(lower_triangular_c), color='red', label='test-test', shade=True)

    # Set the x-label, y-label, and title of the plot
    plt.xlabel('Similarity score')
    plt.ylabel('Density')
    plt.title('Smooth distribution of Tanimoto similarity scores')

    # Set the legend
    plt.legend()

    # Save the figure
    plt.savefig('Tanimoto Similarity_Distribution_combined.png')
