import tensorflow as tf
import matplotlib.pylab as plt
import os
import numpy as np
import pdb
import copy
import pickle
import sys
import argparse
sys.path.insert(0, "../")
import models
import util

#----------------------------
# set parameters

# get options
parser = util.parser_run()
parser.add_argument('-nItemMax', type=int, default=5, help='maximum number of items in a set, default=5')
args = parser.parse_args()

# mode name
mode = util.mode_name(args.mode)

# maximum value of digits
max_value = 5

# number of epochs
epochs = 200

# early stoppoing parameter
patience = 10

# batch size
batch_size = 20

# number of train data
num_train = 5000

# number of representive vectors
rep_vec_num = 1

# set random seed
os.environ['TF_DETERMINISTIC_OPS'] = '1'
np.random.seed(args.trial)
tf.random.set_seed(args.trial)
#----------------------------

#----------------------------
# make path
experimentPath = 'experiment'
if not os.path.exists(experimentPath):
    os.makedirs(experimentPath)

# make CNN path
cnnModelPath = os.path.join(experimentPath, f'pretrainCNN_{args.baseChn}')
cnnModelPath = os.path.join(cnnModelPath,f"nItemMax{args.nItemMax}_maxValue{max_value}")

if not os.path.exists(cnnModelPath):
    path = os.path.join(cnnModelPath,'model')
    os.makedirs(path) 

    path = os.path.join(cnnModelPath,'result')
    os.makedirs(path)

# make set-to-set model path
modelPath = os.path.join(experimentPath, f'{mode}_{args.baseChn}')

if args.is_set_norm:
    modelPath+=f'_setnorm'

if args.is_cross_norm:
    modelPath+=f'_crossnorm' 

modelPath = os.path.join(modelPath,f"nItemMax{args.nItemMax}_maxValue{max_value}")
modelPath = os.path.join(modelPath,f"layer{args.num_layers}")
modelPath = os.path.join(modelPath, f'num_head{args.num_heads}')
modelPath = os.path.join(modelPath,f"{args.trial}")
if not os.path.exists(modelPath):
    path = os.path.join(modelPath,'model')
    os.makedirs(path)

    path = os.path.join(modelPath,'result')
    os.makedirs(path)
#----------------------------

#----------------------------
# make data
dataFileName = f'pickle_data/MNIST_eventotal_{args.nItemMax}_{max_value}.pkl'
with open(dataFileName,'rb') as fp:
    x_train=pickle.load(fp)
    y_train=pickle.load(fp)
    x_size_train=pickle.load(fp)
    x_test=pickle.load(fp)
    y_test=pickle.load(fp)
    x_test_size=pickle.load(fp)
x_size_train = x_size_train.astype(np.float32)
x_test_size = x_test_size.astype(np.float32)

x_train = x_train[:num_train]
y_train = y_train[:num_train]
x_size_train = x_size_train[:num_train]

# image size (height, weidth, channels)
H, W, C = 28, 28, 1

# normalize images
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# reshape to（N，H，W，C）
x_train = x_train.reshape(x_train.shape[0], args.nItemMax, H, W, C)
x_test = x_test.reshape(x_test.shape[0], args.nItemMax, H, W, C)
#----------------------------

#----------------------------
# CNN 
model_cnn = models.CNN(baseChn=args.baseChn)

# checkpoint and earlystopping
cnn_checkpoint_path = os.path.join(cnnModelPath,"model/cp.ckpt")
cnn_checkpoint_dir = os.path.dirname(cnn_checkpoint_path)
cnn_cp_callback = tf.keras.callbacks.ModelCheckpoint(cnn_checkpoint_path, monitor='val_accuracy', save_weights_only=True, mode='max', save_best_only=True, save_freq='epoch', verbose=1)
cnn_cp_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, mode='max', min_delta=0, verbose=1)
cnn_result_path = os.path.join(cnnModelPath,"result/result.pkl")

if not os.path.exists(cnn_result_path):
    print("train cnn model")

    # setting training, loss, metric to model
    model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)

    # execute pretraining
    history = model_cnn.fit((x_train,x_size_train), y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[cnn_cp_callback, cnn_cp_earlystopping])

    # accuracy and loss
    cnn_acc = history.history['accuracy']
    cnn_val_acc = history.history['val_accuracy']
    cnn_loss = history.history['loss']
    cnn_val_loss = history.history['val_loss']

    # plot loss & acc
    util.plotLossACC(cnnModelPath,cnn_loss,cnn_val_loss,cnn_acc,cnn_val_acc)

    # dump to pickle
    with open(cnn_result_path,'wb') as fp:
        pickle.dump(cnn_acc,fp)
        pickle.dump(cnn_val_acc,fp)
        pickle.dump(cnn_loss,fp)
        pickle.dump(cnn_val_loss,fp)
#----------------------------

#----------------------------
# set-matching network
model_smn = models.SMN(is_final_linear=False, is_set_norm=args.is_set_norm, is_cross_norm=args.is_cross_norm, num_layers=args.num_layers, num_heads=args.num_heads, baseChn=args.baseChn, mode=mode, rep_vec_num=rep_vec_num)

# load CNN
print("load cnn model")
model_cnn.load_weights(cnn_checkpoint_path)
model_smn.CNN = model_cnn

# checkpoint and earlystopping
checkpoint_path = os.path.join(modelPath,"model/cp.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_binary_accuracy', save_weights_only=True, mode='max', save_best_only=True, save_freq='epoch', verbose=1)
cp_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=patience, mode='max', min_delta=0.001, verbose=1)
result_path = os.path.join(modelPath,"result/result.pkl")

if not os.path.exists(result_path):

    # setting training, loss, metric to model
    model_smn.compile(optimizer="adam",loss='binary_crossentropy',metrics=['binary_accuracy'],run_eagerly=True)

    # execute training
    history = model_smn.fit((x_train,x_size_train),y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True, callbacks=[cp_callback,cp_earlystopping])

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
    model_cnn.load_weights(cnn_checkpoint_path)
    model_smn.CNN = model_cnn
    model_smn.load_weights(checkpoint_path)
#---------------------------------

#---------------------------------
# calc test loss and accuracy, and save to pickle
test_loss_path = os.path.join(modelPath, "result/test_loss_acc.txt") 

if not os.path.exists(test_loss_path):
    model_smn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['binary_accuracy'],run_eagerly=True)
    test_loss, test_acc = model_smn.evaluate((x_test,x_test_size),y_test,batch_size=batch_size,verbose=0)

    with open(test_loss_path,'w') as fp:
        fp.write('test loss:' + str(test_loss) + '\n')
        fp.write('test accuracy:' + str(test_acc) + '\n')

    path = os.path.join(modelPath, "result/test_loss_acc.pkl")        
    with open(path,'wb') as fp:
        pickle.dump(test_loss,fp)
        pickle.dump(test_acc,fp)
#----------------------------

#----------------------------
# analysis of set-rep vector
hist_path = f'{modelPath}/result/item_similarity_hist.png'
if not os.path.exists(hist_path):
    num_set = 100

    # predict
    _, predSMN, state = model_smn((x_test[:num_set],x_test_size[:num_set]))
    y_labels = model_smn.cross_set_label(y_test[:num_set])


    #---------------------
    # item vectors

    # get item vectors at each layer in encoder and decoder
    if mode.find('setRepVec') > -1:
        x_enc_item = np.stack([state[f'x_encoder_layer_{l}'][:,:,1:] for l in range(args.num_layers+1)])
        x_dec_item = np.stack([state[f'x_decoder_layer_{l}'][:,:,1:] for l in range(1,args.num_layers+1)])
    else:
        x_enc_item = np.stack([state[f'x_encoder_layer_{l}'] for l in range(args.num_layers+1)])
        x_dec_item = np.stack([state[f'x_decoder_layer_{l}'] for l in range(1,args.num_layers+1)])    

    x_item = np.vstack([x_enc_item,x_dec_item])

    # mean similarity between cross items
    mean_dot_item = np.stack([[[np.sum(np.dot(x_item[k,i,j],x_item[k,j,i].T)/(np.dot(np.linalg.norm(x_item[k,i,j],axis=1,keepdims=1),np.linalg.norm(x_item[k,j,i],axis=1,keepdims=1).T)+1.0e-8))/(x_test_size[i]*x_test_size[j]) for i in range(num_set)] for j in range(num_set)] for k in range(len(x_item))])
    mean_dot_item_pos = np.array([mean_dot_item[i][np.where(y_labels==1)] for i in range(len(x_item))])
    mean_dot_item_neg = np.array([mean_dot_item[i][np.where(y_labels==0)] for i in range(len(x_item))])

    # plot histogram
    util.plotHist(mean_dot_item_pos,mean_dot_item_neg, mode=mode, fname=hist_path)
    #---------------------

    #---------------------
    # set-rep vectors

    # compute set-rep vector at each layer in encoder and decoder
    if mode.find('setRepVec') > -1 or mode == 'CSS':
        x_enc_rep = np.stack([state[f'x_encoder_layer_{l}'][:,:,0] for l in range(args.num_layers+1)])
        x_dec_rep = np.stack([state[f'x_decoder_layer_{l}'][:,:,0] for l in range(args.num_layers+1)])
    elif mode == 'maxPooling':
        x_enc_rep = np.stack([np.max(state[f'x_encoder_layer_{l}'],axis=2) for l in range(args.num_layers+1)])
        x_dec_rep = np.stack([np.max(state[f'x_decoder_layer_{l}'],axis=2) for l in range(args.num_layers+1)])
    elif mode == 'poolingMA':
        set_emb_tile = tf.tile(model_smn.set_emb, [num_set,num_set,1,1])
        x_enc_rep = np.stack([model_smn.pma(set_emb_tile,state[f'x_encoder_layer_{l}']) for l in range(args.num_layers+1)])[:,:,:,0]
        x_dec_rep = np.stack([model_smn.pma(set_emb_tile,state[f'x_decoder_layer_{l}']) for l in range(args.num_layers+1)])[:,:,:,0]
    else:
        x_enc_rep = []
        x_dec_rep = []

    # similarity between set-rep vectors and plot histogram
    if len(x_enc_rep):
        x_rep = np.vstack([x_enc_rep,x_dec_rep])

        # similarity between two set-rep vectors in positive and negative class    
        dot_rep = np.array(np.array([[[np.dot(x_rep[k,i,j],x_rep[k,j,i])/(np.linalg.norm(x_rep[k,i,j])*np.linalg.norm(x_rep[k,j,i])) for i in range(num_set)] for j in range(num_set)] for k in range(len(x_rep))]))
        dot_rep_pos = np.array([dot_rep[i][np.where(y_labels==1)] for i in range(len(x_rep))])
        dot_rep_neg = np.array([dot_rep[i][np.where(y_labels==0)] for i in range(len(x_rep))])

        # plot histograms
        util.plotHist(dot_rep_pos,dot_rep_neg, mode=mode, fname=f'{modelPath}/result/rep_similarity_hist.png')
    #---------------------
#----------------------------


