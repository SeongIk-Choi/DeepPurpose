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

        #if mode not in ['standard', 'streamlit']:
        #    raise Exception('Invalid mode value for the BaseWorker (select between standard and streamlit)')
        #else:
        #    self.mode = mode

    def compute(self, budget, config):
        '''The input parameter 'config' (dictionary) contains the sampled configurations passed by the bohb optimizer
        '''
        current_config = self.base_config.copy()
        temp_config = dict(config)
        original_budget = int(budget)
        current_time = datetime.now().strftime('%d_%m_%Y__%H_%M_%S')
        print("current_config:", current_config)
        print("tmp_config:", temp_config)
        #print("original_budget:", original_budget)
        
        # isolate the instance and target specific parameters that are stored in specific instance_branch_params and target_branch_params nested directories
        #drug_specific_param_keys = [p_name for p_name in temp_config.keys() if p_name.startswith('drug_') and p_name not in ['drug_branch_architecture']]
        #target_specific_param_keys = [p_name for p_name in temp_config.keys() if p_name.startswith('target_') and p_name not in ['target_branch_architecture']]

        #current_config['drug_branch_params'] = {p_name: temp_config[p_name] for p_name in drug_specific_param_keys}
        #current_config['target_branch_params'] = {p_name: temp_config[p_name] for p_name in target_specific_param_keys}
        #current_config.update({p_name: temp_config[p_name] for p_name in temp_config.keys() if p_name not in drug_specific_param_keys+target_specific_param_keys})
        #current_config['results_path'] = self.project_name
        #current_config['experiment_name'] = current_time
        #current_config = generate_config_hp(**current_config)

        self.older_model_dir = None
        self.older_model_budget = None

        current_config.update(
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
                'config': current_config,
            }

        # update the actual budget that will be used to train the model
        current_config.update({'num_epochs': int(budget), 'actuall_budget': int(budget)})
        #print("before_change:",current_config)
        current_config.update(temp_config) # It updates all of the dictionary value that has same keys between tmp_config and current config. 
        print("updated_config:", current_config)
        if current_config['general_architecture_version'] == 'dot_product' :
            if current_config['drug_encoding'] != 'Transformer' and current_config['target_encoding'] == 'Transformer':
                current_config['transformer_emb_size_target'] = current_config['hidden_dim_drug']
                while int(current_config['transformer_emb_size_target']) % int(current_config['transformer_num_attention_heads_target']) != 0:
                    current_config['transformer_num_attention_heads_target'] = current_config['transformer_num_attention_heads_target'] + 1 

            elif current_config['drug_encoding'] == 'Transformer' and current_config['drug_encoding'] != 'Transformer':
                current_config['transformer_emb_size_drug'] = current_config['hidden_dim_target']
                while int(current_config['transformer_emb_size_durg']) % int(current_config['transformer_num_attention_heads_drug']) != 0:
                    current_config['transformer_num_attention_heads_drug'] = current_config['transformer_num_attention_heads_drug'] + 1 
            
        else:
            if current_config['drug_encoding'] != 'Transformer' and current_config['target_encoding'] == 'Transformer':
                #current_config['transformer_emb_size_target'] = current_config['hidden_dim_drug']
                while int(current_config['transformer_emb_size_target']) % int(current_config['transformer_num_attention_heads_target']) != 0:
                    current_config['transformer_num_attention_heads_target'] = current_config['transformer_num_attention_heads_target'] + 1 

            elif current_config['drug_encoding'] == 'Transformer' and current_config['drug_encoding'] != 'Transformer':
                #current_config['transformer_emb_size_drug'] = current_config['hidden_dim_target']
                while int(current_config['transformer_emb_size_durg']) % int(current_config['transformer_num_attention_heads_drug']) != 0:
                    current_config['transformer_num_attention_heads_drug'] = current_config['transformer_num_attention_heads_drug'] + 1 
            
 


        #print("after_change:",current_config)
        #current_config['LR'] = temp_config['LR']
        #current_config['transformer_emb_size_target'] = temp_config['transformer_emb_size_target']
        #current_config['transformer_n_layer_target'] = temp_config['transformer_n_layer_target']
        #print("current_config_update:", current_config)

        #if config['general_architecture_version'] == ~~~~ 
        print("final_config:", current_config)
        # initialize a new model or continue training from an older version with the same configuration
        if len(self.config_to_model[model_config_key]['model_dir']) != 0:
            model = models.model_initialize(**current_config, checkpoint_dir=self.older_model_dir)
            #if self.mode == 'standard':    
            #else:
            #    model = DeepMTP_st(config=current_config, checkpoint_dir=self.older_model_dir)
        else:
            model = models.model_initialize(**current_config)
            #if self.mode == 'standard':   
            #else:
            #    model = DeepMTP_st(config=current_config)

        # train, validate and test all at once
        #print("-------------------train_val_start----------------")
        val_results = model.train(self.train, self.val, self.test)

        #print("config_to_model_pre:", self.config_to_model)
        # append all the latest relevant info for the given configuration
        self.config_to_model[model_config_key]['budget'].append(original_budget)
        self.config_to_model[model_config_key]['model_dir'].append(self.project_name+current_time+'/model.pt')
        #self.config_to_model[model_config_key]['model_dir'].append(self.project_name+current_time+'/model.pt')
        self.config_to_model[model_config_key]['run_name'].append(current_time)
        #print("config_to_model:", self.config_to_model)
        #print("check:",self.config_to_model[model_config_key]['model_dir'])
        #print("check2:",self.config_to_model[model_config_key]['model_dir'][-1])

        #print("config_to_model_aft:", self.config_to_model)
        # output_file = open(self.project_name+'/'+current_time+ '/run_results.pkl', 'wb')
        # pickle.dump(self.config_to_model, output_file)
        # output_file.close()

        # output_file = open('hyperopt/older_models/'+wandb_run_project_name+'/run_results.json', 'w')
        # json.dump(self.config_to_model, output_file)
        # output_file.close()
        #print("val_results:", str(min(val_results)))
        #print("optimize:", self.optimize)
        #print("loss_list:", val_results['loss'])
        print("모델확인:",self.config_to_model[model_config_key])
        return {
            'loss': str(min(val_results)),  # remember: always minimizes!
            'info': {'model_dir': self.config_to_model[model_config_key]['model_dir'][-1], 'config': current_config},
        }

        # return {
        #     'loss': val_results['val_'+self.optimize if 'val' not in self.optimize else self.optimize],  # remember: always minimizes!
        #     'info': {'model_dir': self.config_to_model[model_config_key]['model_dir'][-1], 'config': current_config},
        # }
