from pickletools import optimize
import sys
import sys
import torch
sys.path.insert(0, '../../../..')
import os
from datetime import datetime
#from DeepMTP.main import DeepMTP
#from DeepMTP.main_streamlit import DeepMTP as DeepMTP_st
#from DeepMTP.utils.utils import generate_config
from DeepPurpose.utils import *
from DeepPurpose import DTI as models

class BaseWorker:
    ''' Implements a basic worker that can be used by HPO methods. The basic idea is that an HPO methods just has to pass a config and then it gets back the performance of the best epoch on the validation set
    '''
    def __init__(
        self, train, val, test, base_config, metric_to_optimize
    ):
        if not os.path.exists(base_config['hpo_results_path']):
            os.mkdir(base_config['hpo_results_path'])

        self.project_name = base_config['hpo_results_path']+datetime.now().strftime('%d_%m_%Y__%H_%M_%S')+'/'
        if not os.path.exists(self.project_name):
            os.mkdir(self.project_name)

        self.config_to_model = {}
        self.older_model_dir = None
        self.current_model_dir = None
        self.older_model_budget = None
        self.older_model = None
        self.optimize = metric_to_optimize

        self.train = train
        self.val = val
        self.test = test

        self.base_config = base_config


    def compute(self, budget, config):
        '''The input parameter 'config' (dictionary) contains the sampled configurations passed by the bohb optimizer
        '''
        current_config = self.base_config.copy()
        temp_config = dict(config)
        original_budget = int(budget)
        current_time = datetime.now().strftime('%d_%m_%Y__%H_%M_%S')
        self.older_model_dir = None
        self.older_model_budget = None
        self.report_config = None

        final_config = current_config.copy()
        final_config.update(
            {'budget': budget, 'budget_int': int(budget)}
        )
        # create a key from the given configuration
        model_config_key = tuple(sorted(temp_config.items()))
        # check if the configuration has already been seeen. If so extract relevant info from the last experiment with that configuration
        if model_config_key in self.config_to_model:
            self.older_model_dir = self.config_to_model[model_config_key]['model_dir'][-1]
            self.older_model_budget = self.config_to_model[model_config_key]['budget'][-1]
            budget = budget - self.older_model_budget
        else:
            self.config_to_model[model_config_key] = {
                'budget': [],
                'model_dir': [],
                'run_name': [],
                'config': final_config,
            }
        MLP_drug_list = ['Morgan', 'ErG', 'Pubchem', 'Daylight', 'rdkit_2d_normalized', 'ESPF'] 
        MLP_target_list = ['ACC', 'Pseudo AAC', 'Conjoint_triad', 'Quasi-seq', 'ESPF']
        # update the actual budget that will be used to train the model
        final_config.update({'num_epochs': int(budget), 'actuall_budget': int(budget), 'train_epoch': int(budget)})
        final_config.update(temp_config) # It updates all of the dictionary value that has same keys between tmp_config and current config. 
        if final_config['general_architecture_version'] == 'mlp' :
            if final_config['drug_encoding'].startswith('CNN') or final_config['target_encoding'].startswith('CNN') :
                update_cnn_drug_filters = [int(temp_config['cnn_drug_filter_number']), int(temp_config['cnn_drug_filter_number'])*2, int(temp_config['cnn_drug_filter_number'])*3]
                update_cnn_drug_kernels = [int(temp_config['cnn_drug_kernel_number']), int(temp_config['cnn_drug_kernel_number'])*2, int(temp_config['cnn_drug_kernel_number'])*3]
                update_cnn_target_filters = [int(temp_config['cnn_target_filter_number']), int(temp_config['cnn_target_filter_number'])*2, int(temp_config['cnn_target_filter_number'])*3]
                update_cnn_target_kernels = [int(temp_config['cnn_target_kernel_number']), int(temp_config['cnn_target_kernel_number'])*2, int(temp_config['cnn_target_kernel_number'])*3]
                final_config.update({'cnn_drug_filters': update_cnn_drug_filters, 'cnn_drug_kernels': update_cnn_drug_kernels, 'cnn_target_filters':update_cnn_target_filters, 'cnn_target_kernels': update_cnn_target_kernels})

            if final_config['drug_encoding'] in MLP_drug_list or final_config['target_encoding'] in MLP_target_list :
                update_mlp_hidden_dim_drug = [int(temp_config['mlp_hidden_dim_drug_number']), int(temp_config['mlp_hidden_dim_drug_number']/4), int(temp_config['mlp_hidden_dim_drug_number']/16)]
                update_mlp_hidden_dim_target = [int(temp_config['mlp_hidden_dim_target_number']), int(temp_config['mlp_hidden_dim_target_number']/4), int(temp_config['mlp_hidden_dim_target_number']/16)]
                final_config.update({'mlp_hidden_dims_drug': update_mlp_hidden_dim_drug, 'mlp_hidden_dims_target':update_mlp_hidden_dim_target})

            if final_config['target_encoding'] == 'Transformer':
                final_config['hidden_dim_protein'] = final_config['transformer_emb_size_target']
                while int(final_config['transformer_emb_size_target']) % int(final_config['transformer_num_attention_heads_target']) != 0:
                    final_config['transformer_num_attention_heads_target'] = int(final_config['transformer_num_attention_heads_target']) + 1 
                    if int(final_config['transformer_num_attention_heads_target']) == int(16):
                        final_config['transformer_num_attention_heads_target'] = int(1)
            
            if final_config['drug_encoding'] == 'Transformer':
                final_config['hidden_dim_drug'] = final_config['transformer_emb_size_drug']
                while int(final_config['transformer_emb_size_drug']) % int(final_config['transformer_num_attention_heads_drug']) != 0:
                    final_config['transformer_num_attention_heads_drug'] = int(final_config['transformer_num_attention_heads_drug']) + 1 
                    if int(final_config['transformer_num_attention_heads_drug']) == int(16):
                        final_config['transformer_num_attention_heads_drug'] = int(1)   
            
            update_cls_hidden_dims = [int(temp_config['cls_hidden_dim_number']), int(temp_config['cls_hidden_dim_number']), int(temp_config['cls_hidden_dim_number']/2)]
            
            final_config.update({'cls_hidden_dims': update_cls_hidden_dims})

        # Think about this part (dot_prodcut) later on...  
        if final_config['general_architecture_version'] == 'dot_product' :
            if final_config['drug_encoding'] != 'Transformer' and final_config['target_encoding'] == 'Transformer':
                final_config['transformer_emb_size_target'] = final_config['hidden_dim_protein']
                while int(final_config['transformer_emb_size_target']) % int(final_config['transformer_num_attention_heads_target']) != 0:
                    final_config['transformer_num_attention_heads_target'] = final_config['transformer_num_attention_heads_target'] + 1 

            elif final_config['drug_encoding'] == 'Transformer' and final_config['drug_encoding'] != 'Transformer':
                final_config['transformer_emb_size_drug'] = final_config['hidden_dim_drug']
                while int(final_config['transformer_emb_size_durg']) % int(final_config['transformer_num_attention_heads_drug']) != 0:
                    final_config['transformer_num_attention_heads_drug'] = final_config['transformer_num_attention_heads_drug'] + 1 
                    
        # initialize a new model or continue training from an older version with the same configuration
        if len(self.config_to_model[model_config_key]['model_dir']) != 0:
            checkpoint_dir = self.older_model_dir[:-9]
            net = models.model_pretrained(path_dir = checkpoint_dir)
            net.config.update(final_config)

            val_results = net.train(self.train, self.val, self.test)
            net.save_model('./tmp_model'+'/'+current_time)
            
            self.report_config = net.config

        else:
            model = models.model_initialize(**final_config)
            val_results = model.train(self.train, self.val, self.test)
            model.save_model('./tmp_model'+'/'+current_time)
            self.report_config = final_config

        # train, validate and test all at once
        # append all the latest relevant info for the given configuration
        self.config_to_model[model_config_key]['budget'].append(original_budget)
        self.config_to_model[model_config_key]['model_dir'].append('./tmp_model/' + current_time+'/model.pt')
        self.config_to_model[model_config_key]['run_name'].append(current_time)

        return {
            'loss': str(min(val_results)),  # remember: always minimizes!
            'info': {'model_dir': self.config_to_model[model_config_key]['model_dir'][-1], 'config': self.report_config},
        }