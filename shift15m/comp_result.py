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
args = parser.parse_args()

# year of data, max number of items, number of candidates
year = 2017
max_item_num = 5
test_cand_num = 5

# experimet path
experimentPath = 'experiment'

# split modes and trials with cooma
modes = args.modes.split(',')
num_layers = args.num_layers.split(',')
num_heads = args.num_heads.split(',')
trials = args.trials.split(',')

# add CMC to eval_name
[ eval_name.append(f'CMC={i+1}') for i in range(test_cand_num) ]
eval_cmc_name = []
[ eval_cmc_name.append(f'CMC_mean={i+1}') for i in range(test_cand_num) ]
[ eval_cmc_name.append(f'CMC_std={i+1}') for i in range(test_cand_num) ]

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

                modelPath = os.path.join(modelPath,f"year{year}")
                modelPath = os.path.join(modelPath,f"max_item_num{max_item_num}")
                modelPath = os.path.join(modelPath,f"layer{num_layers[lind]}")
                modelPath = os.path.join(modelPath, f'num_head{num_heads[hind]}')
                modelPath = os.path.join(modelPath,f"{trials[tind]}")
                modelPath = os.path.join(modelPath,"result")

                path = os.path.join(modelPath,f"test_loss_acc.pkl")
                with open(path,'rb') as fp:
                    eval_values.append(pickle.load(fp)) # test_loss
                    eval_values.append(pickle.load(fp)) # test_acc
                    eval_values.extend(pickle.load(fp)) # cmcs
                #---------------------                    

            #---------------------
            # create dataframe
            df_tmp = pd.DataFrame(np.reshape(eval_values,[len(trials),-1]),columns=eval_name)
            df_tmp['trial'] = trials
            df_tmp['mode'] = [mode]*len(trials)
            df_tmp['layer'] = [num_layers[lind]]*len(trials)
            df_tmp['head'] = [num_heads[hind]]*len(trials)
         
            if not (sind+lind+hind):
                df = df_tmp
                #df_cmc = df_cmc_tmp
            else:
                df = pd.concat([df,df_tmp])
                #df_cmc = pd.concat([df_cmc,df_cmc_tmp])

# print results
print(df.groupby(['head','mode','layer'])[ eval_name ].agg(['mean','std']).round(3))
