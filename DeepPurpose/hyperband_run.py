from rdkit import Chem
from DeepPurpose import DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *
from ast import arg
from DeepPurpose.hpo_worker import BaseWorker
from DeepPurpose.simple_hyperband import HyperBand
import torch
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import CategoricalHyperparameter
# define the configuration space
cs= CS.ConfigurationSpace()

LR= CSH.UniformFloatHyperparameter("LR", lower=1e-6, upper=1e-3, default_value=1e-3, log=True)

input_dim_drug=CSH.UniformIntegerHyperparameter("input_dim_drug", lower=512, upper=2048, default_value=1024, log=False)
input_dim_protein=CSH.UniformIntegerHyperparameter("input_dim_protein", lower=512, upper=16840, default_value=8420, log=False)

hidden_dim_drug=CSH.UniformIntegerHyperparameter("hidden_dim_drug", lower=8, upper=2048, default_value=128, log=False)
hidden_dim_protein=CSH.UniformIntegerHyperparameter("hidden_dim_protein", lower=8, upper=2048, default_value=128, log=False)

cls_hidden_dim_number = CSH.UniformIntegerHyperparameter("cls_hidden_dim_number", lower=4, upper=1024, default_value=1024, log=False)

mlp_hidden_dim_drug_number = CSH.UniformIntegerHyperparameter("mlp_hidden_dim_drug_number", lower=4, upper=1024, default_value=1024, log=False)
mlp_hidden_dim_target_number = CSH.UniformIntegerHyperparameter("mlp_hidden_dim_target_number", lower=4, upper=1024, default_value=1024, log=False)

mpnn_hidden_size=CSH.UniformIntegerHyperparameter("mpnn_hidden_size", lower=8, upper=516, default_value=128, log=False)
mpnn_depth=CSH.UniformIntegerHyperparameter("mpnn_depth", lower=2, upper=8, default_value=3, log=False)

rnn_drug_hid_dim=CSH.UniformIntegerHyperparameter("rnn_drug_hid_dim", lower=2, upper=516, default_value=64, log=False)
rnn_target_hid_dim=CSH.UniformIntegerHyperparameter("rnn_target_hid_dim", lower=2, upper=516, default_value=64, log=False)

rnn_drug_n_layers = CSH.UniformIntegerHyperparameter("rnn_drug_n_layers", lower=1, upper=4, default_value=2, log=False)
rnn_target_n_layers = CSH.UniformIntegerHyperparameter("rnn_target_n_layers", lower=1, upper=4, default_value=2, log=False)

cnn_drug_filter_number = CSH.UniformIntegerHyperparameter("cnn_drug_filter_number", lower=4, upper=64, default_value=32, log=False)

cnn_drug_kernel_number = CSH.UniformIntegerHyperparameter("cnn_drug_kernel_number", lower=2, upper=8, default_value=4, log=False)

cnn_target_filter_number = CSH.UniformIntegerHyperparameter("cnn_target_filter_number", lower=4, upper=64, default_value=32, log=False)

cnn_target_kernel_number = CSH.UniformIntegerHyperparameter("cnn_target_kernel_number", lower=2, upper=8, default_value=4, log=False)

gnn_hid_dim_drug = CSH.UniformIntegerHyperparameter("gnn_hid_dim_drug", lower=32, upper=128, default_value=64, log=False)

gnn_num_layers = CSH.UniformIntegerHyperparameter("gnn_num_layers", lower=1, upper=4, default_value=2, log=False)

neuralfp_max_degree = CSH.UniformIntegerHyperparameter("neuralfp_max_degree", lower=2, upper=20, default_value=10, log=False)

neuralfp_predictor_hid_dim = CSH.UniformIntegerHyperparameter("neuralfp_predictor_hid_dim", lower=1, upper=128, default_value=128, log=False)

transformer_emb_size_drug= CSH.UniformIntegerHyperparameter("transformer_emb_size_drug", lower=1, upper=4, default_value=2, log=False)
transformer_emb_size_target= CSH.UniformIntegerHyperparameter("transformer_emb_size_target", lower=1, upper=4, default_value=2, log=False)

transformer_intermediate_size_drug = CSH.UniformIntegerHyperparameter("transformer_intermediate_size_drug", lower=1, upper = 4, default_value =2, log=False)
transformer_intermediate_size_target = CSH.UniformIntegerHyperparameter("transformer_intermediate_size_target", lower=1, upper = 4, default_value =2, log=False)

transformer_n_layer_drug= CSH.UniformIntegerHyperparameter("transformer_n_layer_drug", lower=1, upper=2, default_value=1, log=False)
transformer_n_layer_target= CSH.UniformIntegerHyperparameter("transformer_n_layer_target", lower=1, upper=2, default_value=1, log=False)

transformer_dropout_rate = CSH.UniformFloatHyperparameter("transformer_dropout_rate", lower=0.1, upper=0.5, default_value=0.1, log=True)


cs.add_hyperparameters(
    [
        LR,
        input_dim_drug,
        input_dim_protein,
        hidden_dim_drug,
        hidden_dim_protein,
        mpnn_hidden_size, 
        mpnn_depth,
        cls_hidden_dim_number,
        mlp_hidden_dim_drug_number,
        mlp_hidden_dim_target_number,
        rnn_drug_hid_dim,
        rnn_target_hid_dim,
        rnn_drug_n_layers,
        transformer_intermediate_size_drug,
        rnn_target_n_layers,
        cnn_drug_filter_number,
        cnn_drug_kernel_number,
        cnn_target_filter_number,
        cnn_target_kernel_number,
        gnn_hid_dim_drug,
        transformer_intermediate_size_target,
        gnn_num_layers,
        neuralfp_max_degree,
        neuralfp_predictor_hid_dim,
        transformer_emb_size_drug,
        transformer_n_layer_drug,
        transformer_emb_size_target,
        transformer_n_layer_target,
        transformer_dropout_rate,
    ]
)

X_drug, X_target, y = load_process_DAVIS('./data/', binary=False)

drug_encoding = 'CNN'
target_encoding = 'CNN'
train, val, test = data_process(X_drug, X_target, y,
                                drug_encoding, target_encoding,
                                split_method='random',frac=[0.7,0.1,0.2])


general_architecture_version='mlp'
additional_info = {'eta': 3, 'max_budget': 81}
config = generate_config(drug_encoding = drug_encoding,
                         target_encoding = target_encoding,
                         general_architecture_version = general_architecture_version,
                         cls_hidden_dims = [1024,1024,512],
                         train_epoch = 10,
                         test_every_X_epoch = 2,
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
                         wandb_project_name="verification2",
                         wandb_project_entity="seongik-choi",
                         hpo_results_path='/kyukon/data/gent/vo/000/gvo00048/vsc44416/hyperband_verification/',
                         rnn_target_hid_dim=64,
                         result_folder = "/kyukon/data/gent/vo/000/gvo00048/vsc44416/result_verification/",
                         use_early_stopping = True,
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
    verbose=True
)

# start-up the optimizer
best_overall_config = hb.run_optimizer()