import os 
aff = os.sched_getaffinity(0)

from rdkit import Chem
from DeepPurpose import DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *
from ast import arg
from DeepPurpose.hpo_worker import BaseWorker
from DeepPurpose.simple_hyperband import HyperBand
import torch
os.sched_setaffinity(0,aff)
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import CategoricalHyperparameter

# Generate train, val, test dataset
X_drug, X_target, y = load_process_KIBA('./data/', binary=False)

drug_encoding = 'Morgan'
target_encoding = 'AAC'
train, val, test = data_process(X_drug, X_target, y,
                                drug_encoding, target_encoding,
                                split_method='cold_drug_protein_interaction',frac=[0.7,0.1,0.2])

MLP_drug_list = ['Morgan', 'ErG', 'Pubchem', 'Daylight', 'rdkit_2d_normalized', 'ESPF'] 
MLP_target_list = ['AAC', 'PseudoAAC', 'Conjoint_triad', 'Quasi-seq', 'ESPF']
esm2_model_list = ['esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t33_650M_UR50D']
# define the configuration space
cs= CS.ConfigurationSpace()

LR= CSH.UniformFloatHyperparameter("LR", lower=1e-6, upper=1e-3, default_value=1e-3, log=True)
cls_hidden_dim_number = CSH.UniformIntegerHyperparameter("cls_hidden_dim_number", lower=4, upper=1024, default_value=1024, log=False)
batch_size = CSH.CategoricalHyperparameter("batch_size", [32,64,128,256])
depth = CSH.UniformIntegerHyperparameter("depth", lower=1, upper=4, default_value=3, log=False)
cs.add_hyperparameters([LR, cls_hidden_dim_number, batch_size, depth,])

if drug_encoding.startswith('CNN') :
    cnn_drug_filter_number = CSH.UniformIntegerHyperparameter("cnn_drug_filter_number", lower=4, upper=64, default_value=32, log=False)

    cnn_drug_kernel_number = CSH.UniformIntegerHyperparameter("cnn_drug_kernel_number", lower=2, upper=8, default_value=4, log=False)
    hidden_dim_drug=CSH.UniformIntegerHyperparameter("hidden_dim_drug", lower=8, upper=516, default_value=128, log=False)

    if drug_encoding == 'CNN_RNN' :
        rnn_drug_hid_dim=CSH.UniformIntegerHyperparameter("rnn_drug_hid_dim", lower=2, upper=516, default_value=64, log=False)
        rnn_drug_n_layers = CSH.UniformIntegerHyperparameter("rnn_drug_n_layers", lower=1, upper=2, default_value=2, log=False)
        #hidden_dim_drug=CSH.UniformIntegerHyperparameter("hidden_dim_drug", lower=8, upper=516, default_value=128, log=False)
        cs.add_hyperparameters([cnn_drug_filter_number, cnn_drug_kernel_number, rnn_drug_hid_dim, rnn_drug_n_layers,])
    else:
        cs.add_hyperparameters([cnn_drug_filter_number, cnn_drug_kernel_number, hidden_dim_drug,])


if target_encoding.startswith('CNN'):
    cnn_target_filter_number = CSH.UniformIntegerHyperparameter("cnn_target_filter_number", lower=4, upper=64, default_value=32, log=False)

    cnn_target_kernel_number = CSH.UniformIntegerHyperparameter("cnn_target_kernel_number", lower=2, upper=8, default_value=4, log=False)
    hidden_dim_protein=CSH.UniformIntegerHyperparameter("hidden_dim_protein", lower=8, upper=516, default_value=128, log=False)
    if target_encoding == 'CNN_RNN' :
        rnn_target_hid_dim=CSH.UniformIntegerHyperparameter("rnn_target_hid_dim", lower=2, upper=516, default_value=64, log=False)

        rnn_target_n_layers = CSH.UniformIntegerHyperparameter("rnn_target_n_layers", lower=1, upper=2, default_value=2, log=False)
        cs.add_hyperparameters([cnn_target_filter_number, cnn_target_kernel_number, rnn_target_hid_dim, rnn_target_n_layers,])
    else:
        cs.add_hyperparameters([cnn_target_filter_number, cnn_target_kernel_number, hidden_dim_protein,])


# if drug_encoding == 'CNN' or target_encoding == 'CNN':
#     cnn_drug_filter_number = CSH.UniformIntegerHyperparameter("cnn_drug_filter_number", lower=4, upper=64, default_value=32, log=False)

#     cnn_drug_kernel_number = CSH.UniformIntegerHyperparameter("cnn_drug_kernel_number", lower=2, upper=8, default_value=4, log=False)

#     cnn_target_filter_number = CSH.UniformIntegerHyperparameter("cnn_target_filter_number", lower=4, upper=64, default_value=32, log=False)

#     cnn_target_kernel_number = CSH.UniformIntegerHyperparameter("cnn_target_kernel_number", lower=2, upper=8, default_value=4, log=False)

#     cs.add_hyperparameters([cnn_drug_filter_number, cnn_drug_kernel_number, cnn_target_filter_number, cnn_target_kernel_number,])

if drug_encoding =='Transformer' or target_encoding == 'Transformer': 
    transformer_emb_size_drug= CSH.UniformIntegerHyperparameter("transformer_emb_size_drug", lower=1, upper=4, default_value=2, log=False)
    transformer_emb_size_target= CSH.UniformIntegerHyperparameter("transformer_emb_size_target", lower=1, upper=4, default_value=2, log=False)

    transformer_intermediate_size_drug = CSH.UniformIntegerHyperparameter("transformer_intermediate_size_drug", lower=1, upper = 4, default_value =2, log=False)
    transformer_intermediate_size_target = CSH.UniformIntegerHyperparameter("transformer_intermediate_size_target", lower=1, upper = 4, default_value =2, log=False)

    transformer_n_layer_drug= CSH.UniformIntegerHyperparameter("transformer_n_layer_drug", lower=1, upper=2, default_value=1, log=False)
    transformer_n_layer_target= CSH.UniformIntegerHyperparameter("transformer_n_layer_target", lower=1, upper=2, default_value=1, log=False)

    transformer_dropout_rate = CSH.UniformFloatHyperparameter("transformer_dropout_rate", lower=0.1, upper=0.5, default_value=0.1, log=True)
    #hidden_dim_drug=CSH.UniformIntegerHyperparameter("hidden_dim_drug", lower=8, upper=2048, default_value=128, log=False)
    #hidden_dim_protein=CSH.UniformIntegerHyperparameter("hidden_dim_protein", lower=8, upper=2048, default_value=128, log=False)

    cs.add_hyperparameters([transformer_emb_size_drug, transformer_emb_size_target, transformer_intermediate_size_drug, transformer_intermediate_size_target, transformer_n_layer_drug,
    transformer_n_layer_target, transformer_dropout_rate,])

# if drug_encoding == 'CNN_RNN' or target_encoding == 'CNN_RNN' :     
#     rnn_drug_hid_dim=CSH.UniformIntegerHyperparameter("rnn_drug_hid_dim", lower=2, upper=516, default_value=64, log=False)
#     rnn_target_hid_dim=CSH.UniformIntegerHyperparameter("rnn_target_hid_dim", lower=2, upper=516, default_value=64, log=False)

#     rnn_drug_n_layers = CSH.UniformIntegerHyperparameter("rnn_drug_n_layers", lower=1, upper=2, default_value=2, log=False)
#     rnn_target_n_layers = CSH.UniformIntegerHyperparameter("rnn_target_n_layers", lower=1, upper=2, default_value=2, log=False)
#     hidden_dim_drug=CSH.UniformIntegerHyperparameter("hidden_dim_drug", lower=8, upper=516, default_value=128, log=False)

#     cs.add_hyperparameters([rnn_drug_hid_dim, rnn_target_hid_dim, rnn_drug_n_layers, rnn_target_n_layers, hidden_dim_drug,])


if drug_encoding == 'MPNN' : 

    mpnn_hidden_size=CSH.UniformIntegerHyperparameter("mpnn_hidden_size", lower=8, upper=516, default_value=128, log=False)
    mpnn_depth=CSH.UniformIntegerHyperparameter("mpnn_depth", lower=2, upper=8, default_value=3, log=False)
    hidden_dim_drug=CSH.UniformIntegerHyperparameter("hidden_dim_drug", lower=8, upper=2048, default_value=128, log=False)

    cs.add_hyperparameters([mpnn_hidden_size, mpnn_depth, hidden_dim_drug,])

if drug_encoding.startswith('DGL_') :
    
    gnn_hid_dim_drug = CSH.UniformIntegerHyperparameter("gnn_hid_dim_drug", lower=32, upper=128, default_value=64, log=False)

    gnn_num_layers = CSH.UniformIntegerHyperparameter("gnn_num_layers", lower=1, upper=4, default_value=2, log=False)

    neuralfp_max_degree = CSH.UniformIntegerHyperparameter("neuralfp_max_degree", lower=2, upper=20, default_value=10, log=False)

    neuralfp_predictor_hid_dim = CSH.UniformIntegerHyperparameter("neuralfp_predictor_hid_dim", lower=1, upper=128, default_value=128, log=False)

    cs.add_hyperparameters([gnn_hid_dim_drug, gnn_num_layers, neuralfp_max_degree, neuralfp_predictor_hid_dim,])

if drug_encoding in MLP_drug_list :
    mlp_hidden_dim_drug_number = CSH.UniformIntegerHyperparameter("mlp_hidden_dim_drug_number", lower=4, upper=1024, default_value=1024, log=False)
    cs.add_hyperparameters([mlp_hidden_dim_drug_number,])

if target_encoding in MLP_target_list:
    mlp_hidden_dim_target_number = CSH.UniformIntegerHyperparameter("mlp_hidden_dim_target_number", lower=4, upper=1024, default_value=1024, log=False)
    cs.add_hyperparameters([mlp_hidden_dim_target_number,])

if drug_encoding == 'Conv_CNN_2D' : 
    filter = CSH.CategoricalHyperparameter("filter", [8,16,32,64])
    drop_rate = CSH.CategoricalHyperparameter("drop_rate", [0.2, 0.25, 0.3, 0.4, 0.5])
    fully_layer_1 = CSH.CategoricalHyperparameter("fully_layer_1", [32, 64, 128, 256, 512])
    fully_layer_2 = CSH.CategoricalHyperparameter("fully_layer_2", [32, 64, 128, 256, 512])
    hidden_dim_drug=CSH.UniformIntegerHyperparameter("hidden_dim_drug", lower=8, upper=516, default_value=128, log=False)
    #conv_depth = CSH.UniformIntegerHyperparameter("conv_depth", lower=1, upper=5, default_value=5, log=False)
    cs.add_hyperparameters([filter,drop_rate,fully_layer_1,fully_layer_2,hidden_dim_drug,])

general_architecture_version='mlp'
additional_info = {'eta': 3, 'max_budget': 127, 'split_method': 'cold_drug_protein_interaction', 'dataset': 'KIBA'}
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
                         wandb_project_name="best_model_check",
                         wandb_project_entity="seongik-choi",
                         hpo_results_path='/kyukon/data/gent/vo/000/gvo00048/vsc44416/hyperband/',
                         rnn_target_hid_dim=64,
                         result_folder = "/kyukon/data/gent/vo/000/gvo00048/vsc44416/result_Morgan_AAC_Best_Model_D/",
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


# initialize the BaseWorker that will be used by Hyperband's optimizer

worker = BaseWorker(train, val, test, config, 'loss')
# initialize the optimizers
hb = HyperBand(
    base_worker=worker,
    configspace=cs,
    eta=config['additional_info']['eta'],
    max_budget=config['additional_info']['max_budget'],
    direction='min',
    split_method=config['additional_info']['split_method'],
    verbose=True
)

# start-up the optimizer
best_overall_config = hb.run_optimizer()
