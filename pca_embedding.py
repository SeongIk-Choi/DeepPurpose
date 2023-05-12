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
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import CategoricalHyperparameter
from sklearn.decomposition import PCA
# Generate train, val, test dataset
X_drug, X_target, y = load_process_KIBA('./data/', binary=False)

drug_encoding = 'Morgan'
target_encoding = 'AAC'
train, val, test = data_process(X_drug, X_target, y,
                                drug_encoding, target_encoding,
                                split_method='random',frac=[0.7,0.1,0.2])

general_architecture_version='mlp'
additional_info = {'eta': 3, 'max_budget': 127, 'split_method': 'random', 'dataset': 'KIBA'}
config = generate_config(drug_encoding = drug_encoding,
                         target_encoding = target_encoding,
                         general_architecture_version = general_architecture_version,
                         cls_hidden_dims = [1024,1024,512],
                         train_epoch = 10,
                         test_every_X_epoch = 2,
                         depth=3,
                         LR = 0.001,
                         mlp_hidden_dims_drug = [1024, 256, 64],
                         mlp_hidden_dims_target = [1024, 256, 64],
                         batch_size = 128,
                         hidden_dim_drug = 128,
                         mpnn_hidden_size = 128,
                         mpnn_depth = 3,
                         cnn_target_filters = [32,64,96],
                         cnn_target_kernels = [4,8,12],
                         additional_info=additional_info,
                         cuda_id=0,
                         wandb_project_name="single_train",
                         wandb_project_entity="seongik-choi",
                         hpo_results_path='/kyukon/data/gent/vo/000/gvo00048/vsc44416/hyperband/',
                         rnn_target_hid_dim=64,
                         result_folder = "/kyukon/data/gent/vo/000/gvo00048/vsc44416/",
                         use_early_stopping = True,
                         fully_layer_1 = 256,
                         fully_layer_2 = 128,
                         drop_rate = 0.25,
                         patience = 30,
                         esm_model_dir = "/kyukon/data/gent/vo/000/gvo00048/vsc44416/DeepPurpose/esm_pretrained/",
                         esm_embedding_dir = "/kyukon/data/gent/vo/000/gvo00048/vsc44416/DeepPurpose/esm_embedding/", 
                         fasta_file = "/kyukon/data/gent/vo/000/gvo00048/vsc44416/DeepPurpose/fasta_data/", 
                         include = "mean",
                         num_workers=24,
                         dataset = "KIBA",
                        )

net = models2.model_pretrained('/kyukon/data/gent/vo/000/gvo00048/vsc44416/DeepPurpose/best_model/Morgan_AAC_KIBA_A')

X_drug_pca, X_target_pca = net.convert_to_tensor(test)

