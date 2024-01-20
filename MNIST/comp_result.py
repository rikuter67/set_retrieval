import os
import numpy as np
import pdb
import pickle
import pandas as pd
import argparse
import sys
sys.path.insert(0, "../")
import util

#----------------------------
# set parameters
eval_name = ['test_loss','test_acc']

# get options
parser = util.parser_comp()
parser.add_argument('-nItemMax', type=int, default=5, help='maximum number of items in a set, default=5')
args = parser.parse_args()

# maximum value of digits
max_value = 5

# experimet path
experimentPath = 'experiment'

# split modes and trials with cooma
modes = args.modes.split(',')
num_layers = args.num_layers.split(',')
num_heads = args.num_heads.split(',')
trials = args.trials.split(',')
#----------------------------

for hind in range(len(num_heads)):
    for lind in range(len(num_layers)):
        for sind in range(len(modes)):
            eval_values = []
            for tind in range(len(trials)):
                #---------------------
                # load results
                mode = util.mode_name(int(modes[sind]))
                modelPath = os.path.join(experimentPath, f'{mode}_{args.baseChn}')

                if args.is_set_norm:
                    modelPath+=f'_setnorm' 

                if args.is_cross_norm:
                    modelPath+=f'_crossnorm'                                        

                modelPath = os.path.join(modelPath,f"nItemMax{args.nItemMax}_maxValue{max_value}")
                modelPath = os.path.join(modelPath,f"layer{num_layers[lind]}")
                modelPath = os.path.join(modelPath, f'num_head{num_heads[hind]}')
                modelPath = os.path.join(modelPath,f"{trials[tind]}")
                modelPath = os.path.join(modelPath,"result")

                path = os.path.join(modelPath,"test_loss_acc.pkl")
                with open(path,'rb') as fp:
                    eval_values.append(pickle.load(fp)) # test_loss
                    eval_values.append(pickle.load(fp)) # test_acc
                #---------------------

            #---------------------
            # create dataframe
            df_tmp = pd.DataFrame(np.reshape(eval_values,[len(trials),-1]),columns=eval_name)
            df_tmp['trial'] = trials
            df_tmp['mode'] = [mode]*len(trials)
            df_tmp['layer'] = [num_layers[lind]]*len(trials)
            df_tmp['head'] = [num_heads[hind]]*len(trials)
            
            if not (sind+lind):
                df_trial = df_tmp
            else:
                df_trial = pd.concat([df_trial, df_tmp])
            #---------------------
        
        if not (hind+lind):
            df = df_trial
        else:
            df = df.append(df_trial)

# print results
print(df.groupby(['head','mode','layer'])[['test_loss','test_acc']].agg(['mean','std']).round(3))
