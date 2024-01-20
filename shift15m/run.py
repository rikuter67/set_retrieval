import tensorflow as tf
import matplotlib.pylab as plt
import os
import numpy as np
import pdb
import copy
import pickle
import sys
import argparse
import make_dataset as data
sys.path.insert(0, "../")
import models
import util

#----------------------------
# set parameters

# get options
parser = util.parser_run()
args = parser.parse_args()

# mode name
mode = util.mode_name(args.mode)

# year of data and max number of items
year = 2017
max_item_num = 5
test_cand_num = 5

# number of epochs
epochs = 100

# early stoppoing parameter
patience = 5

# batch size
batch_size = 10

# number of representive vectors
rep_vec_num = 1

# negative down sampling
is_neg_down_sample = True

# set random seed
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(args.trial)
tf.random.set_seed(args.trial)
#----------------------------

#----------------------------
# make Path

# make experiment path containing CNN and set-to-set model
experimentPath = 'experiment'
if not os.path.exists(experimentPath):
    os.makedirs(experimentPath)

# make set-to-set model path
modelPath = os.path.join(experimentPath, f'{mode}_{args.baseChn}')

if args.is_set_norm:
    modelPath+=f'_setnorm'

if args.is_cross_norm:
    modelPath+=f'_crossnorm' 

modelPath = os.path.join(modelPath,f"year{year}")
modelPath = os.path.join(modelPath,f"max_item_num{max_item_num}")
modelPath = os.path.join(modelPath,f"layer{args.num_layers}")
modelPath = os.path.join(modelPath,f"num_head{args.num_heads}")
modelPath = os.path.join(modelPath,f"{args.trial}")
if not os.path.exists(modelPath):
    path = os.path.join(modelPath,'model')
    os.makedirs(path)

    path = os.path.join(modelPath,'result')
    os.makedirs(path)
#----------------------------

#----------------------------
# make data
train_generator = data.trainDataGenerator(year=year, batch_size=batch_size, max_item_num=max_item_num)
x_valid, x_size_valid, y_valid = train_generator.data_generation_val()

# set data generator for test
test_generator = data.testDataGenerator(year=year, cand_num=test_cand_num)
x_test = test_generator.x
x_size_test = test_generator.x_size
y_test = test_generator.y
test_batch_size = test_generator.batch_grp_num
#----------------------------

#----------------------------
# set-matching network
model_smn = models.SMN(isCNN=False, is_final_linear=True, is_set_norm=args.is_set_norm, is_cross_norm=args.is_cross_norm, num_layers=args.num_layers, num_heads=args.num_heads, baseChn=args.baseChn, mode=mode, rep_vec_num=rep_vec_num, is_neg_down_sample=is_neg_down_sample)

checkpoint_path = os.path.join(modelPath,"model/cp.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_binary_accuracy', save_weights_only=True, mode='max', save_best_only=True, save_freq='epoch', verbose=1)
cp_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=patience, mode='max', min_delta=0.001, verbose=1)
result_path = os.path.join(modelPath,"result/result.pkl")

if not os.path.exists(result_path):

    # setting training, loss, metric to model
    model_smn.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'], run_eagerly=True)
    
    # execute training
    history = model_smn.fit(train_generator, epochs=epochs, validation_data=((x_valid, x_size_valid), y_valid), shuffle=True, callbacks=[cp_callback,cp_earlystopping])

    # accuracy and loss
    acc = history.history['binary_accuracy']
    val_acc = history.history['val_binary_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # plot loss & acc
    util.plotLossACC(modelPath,loss,val_loss,acc,val_acc)

    # dump to pickle
    with open(result_path,'wb') as fp:
        pickle.dump(acc,fp)
        pickle.dump(val_acc,fp)
        pickle.dump(loss,fp)
        pickle.dump(val_loss,fp)
else:
    # load trained parameters
    print("load models")
    model_smn.load_weights(checkpoint_path)
#----------------------------

#---------------------------------
# calc test loss and accuracy, and save to pickle
test_loss_path = os.path.join(modelPath, "result/test_loss_acc.txt")
if not os.path.exists(test_loss_path):
    model_smn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['binary_accuracy'],run_eagerly=True)
    test_loss, test_acc = model_smn.evaluate((x_test,x_size_test),y_test,batch_size=test_batch_size,verbose=0)

    # compute cmc
    _, predSMN, _ = model_smn.predict((x_test, x_size_test), batch_size=test_batch_size, verbose=1)
    cmcs = util.calc_cmcs(predSMN, y_test, batch_size=test_batch_size)

    with open(test_loss_path,'w') as fp:
        fp.write('test loss:' + str(test_loss) + '\n')
        fp.write('test accuracy:' + str(test_acc) + '\n')
        fp.write('test cmc:' + str(cmcs) + '\n')

    path = os.path.join(modelPath, "result/test_loss_acc.pkl")
    with open(path,'wb') as fp:
        pickle.dump(test_loss,fp)
        pickle.dump(test_acc,fp)
        pickle.dump(cmcs,fp)
#---------------------------------
