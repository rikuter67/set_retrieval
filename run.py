import os
import sys
import glob
import copy
import argparse
import pickle
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
from pathlib import Path
from PIL import Image
import pdb

import models
import util
import make_dataset as data

#---------------------------------------------------------------------------------------
def parser_run(): # parser for run.py
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, choices=['maxPooling', 'poolingMA', 'CSS', 'setRepVec_biPMA', 'setRepVec_pivot'], default='setRepVec_pivot', help='mode of computing set-matching score')
    parser.add_argument('-baseChn', type=int, default=32, help='number of base channel, default=32') # 次元数の半分？
    parser.add_argument('-model', type=str, default='VLAD', choices=['VLAD', 'SMN', 'random', 'CNN'], help='model type, default=VLAD')
    parser.add_argument('-num_layers', type=int, default=3, help='number of layers (attentions) in encoder and decoder, default=3')
    parser.add_argument('-num_heads', type=int, default=5, help='number of heads in attention, default=5')
    parser.add_argument('-is_set_norm', type=int, default=1, help='switch of set-normalization (1:on, 0:off), default=1')
    parser.add_argument('-is_cross_norm', type=int, default=1, help='switch of cross-normalization (1:on, 0:off), default=1')
    parser.add_argument('-trial', type=int, default=1, help='index of trial, default=1')
    parser.add_argument('-calc_set_sim', type=str, choices=['CS', 'BERTscore'], default='CS', help='how to evaluate set similarity')
    parser.add_argument('-use_Cvec', type=bool, default=True, help='Whether use Cvec')
    parser.add_argument('-is_Cvec_linear', type=bool, default=False, help='Whether learn FC_projection for Cluster seed vec') # 4096次元の候補ベクトルの次元削減FC層を学習するか否か？
    parser.add_argument('-year', type=int, default=2017, help='year of data, default=2017')
    parser.add_argument('-max_item_num', type=int, default=5, help='max number of items, default=5')
    parser.add_argument('-test_cand_num', type=int, default=5, help='number of test candidates, default=5')
    parser.add_argument('-epochs', type=int, default=100, help='number of epochs, default=100')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate, default=0.001')
    parser.add_argument('-patience', type=int, default=5, help='early stopping patience, default=5')
    parser.add_argument('-batch_size', type=int, default=50, help='batch size, default=50')
    parser.add_argument('-rep_vec_num', type=int, default=41, help='number of representative vectors, default=41')
    parser.add_argument('-is_neg_down_sample', type=bool, default=True, help='negative down sampling, default=True')
    parser.add_argument('-pretrained_mlp', type=int, default=0, help='Whether pretrain MLP (not use FC_projection)')
    parser.add_argument('-mlp_projection_dim', type=int, default=128, help='MLP will be learned to achieve designated dimention')
    parser.add_argument('-train', type=bool, default=True, help='Whether train the model')
    return parser
#---------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------
def parser_comp(): # parser for comp_results.py
    parser = argparse.ArgumentParser(description='MNIST eventotal matching')
    parser.add_argument('-modes', default='3,4', help='list of score modes, maxPooling:0, poolingMA:1, CSS:2, setRepVec_biPMA:3, setRepVec_pivot:4, default:3,4')
    parser.add_argument('-baseChn', type=int, default=32, help='number of base channel, default=32')
    parser.add_argument('-num_layers', default='3', help='list of numbers of layers (attentions) in encoder and decoder, default=3')
    parser.add_argument('-num_heads', default='5', help='list of numbers of heads in attention, default=5')
    parser.add_argument('-is_set_norm', type=int, default=1, help='switch of set-normalization (1:on, 0:off), default=1')
    parser.add_argument('-is_cross_norm', type=int, default=1, help='switch of cross-normalization (1:on, 0:off), default=1')
    parser.add_argument('-trials', default='1,2,3', help='list of indices of trials, default=1,2,3')
    parser.add_argument('-calc_set_sim', type=int, default=0, help='how to evaluate set similarity, CS:0, BERTscore:1, default=0')
    return parser
#---------------------------------------------------------------------------------------

parser = parser_run()
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # choose GPU
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

np.random.seed(args.trial)
tf.random.set_seed(args.trial)

# load init seed vectors
pickle_path = "pickle_data/2017-2017-split0"
seed_vectors, args.rep_vec_num = util.load_init_seed_vectors(pickle_path, args.use_Cvec)
# util.plot_3d_tsne(seed_vectors, output_filename="seed_vectors_tsne.png") 

# make Path
experimentPath = 'experiment'

file_name = f"{args.mode}_{args.baseChn}_" + \
    f"model_{args.model}_{'setnorm' if args.is_set_norm else ''}_" + \
    f"{'crossnorm' if args.is_cross_norm else ''}_year{args.year}_maxitem{args.max_item_num}_" + \
    f"layer{args.num_layers}_heads{args.num_heads}_trial{args.trial}_useCvec_{args.use_Cvec}_" + \
    f"isCvecLinear_{args.is_Cvec_linear}_calcSetSim_{args.calc_set_sim}_trainMLP_{args.pretrained_mlp}"


main_model_path = f"{experimentPath}_Train{args.model}"

baseMLPChn = args.mlp_projection_dim * 4
if args.pretrained_mlp:
    mlp_model_path = f"{experimentPath}_TrainMLP"
    
    if not os.path.exists(mlp_model_path):
        os.makedirs(mlp_model_path)
    
    mlp_modelPath = os.path.join(mlp_model_path, file_name)

    # Create directories if they do not exist
    if not os.path.exists(mlp_modelPath):
        os.makedirs(os.path.join(mlp_modelPath, 'model'))
        os.makedirs(os.path.join(mlp_modelPath, 'result'))

    # make data
    train_generator = data.trainDataGenerator(year=args.year, batch_size=args.batch_size, max_item_num=args.max_item_num, mlp_flag=args.pretrained_mlp)
    x_train, y_train = train_generator.data_generation_train()
    x_valid, y_valid = train_generator.data_generation_val()
    
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = {i: total_samples / count for i, count in enumerate(class_counts) if count > 0}
    baseMLPChn = args.mlp_projection_dim * 4
    # Multi layer perceptron model (pretrained network)
    model_mlp = models.MLP(baseChn=baseMLPChn, category_class_num=len(seed_vectors))

    mlp_path = os.path.join(mlp_modelPath)
    mlp_checkpoint_path = os.path.join(mlp_path,"model/cp.ckpt")
    mlp_checkpoint_dir = os.path.dirname(mlp_checkpoint_path)
    mlp_cp_callback = tf.keras.callbacks.ModelCheckpoint(mlp_checkpoint_path, monitor='val_accuracy', save_weights_only=True, mode='max', save_best_only=True, save_freq='epoch', verbose=1)
    mlp_cp_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=args.patience, mode='max', min_delta=0.001, verbose=1)
    mlp_result_path = os.path.join(mlp_path,"result/result.pkl")

    if not os.path.exists(mlp_result_path):
        # setting training, loss, metric to model
        model_mlp.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)
        # execute training
        history = model_mlp.fit(train_generator, epochs=args.epochs, validation_data=(x_valid, y_valid), shuffle=True, class_weight=class_weights, callbacks=[mlp_cp_callback,mlp_cp_earlystopping])

        # accuracy and loss
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # plot loss & acc
        util.plotLossACC(mlp_path,loss,val_loss,acc,val_acc)

        # dump to pickle
        with open(mlp_result_path,'wb') as fp:
            pickle.dump(acc,fp)
            pickle.dump(val_acc,fp)
            pickle.dump(loss,fp)
            pickle.dump(val_loss,fp)

    else:
        # load trained parameters
        print("load models")
        model_mlp.load_weights(mlp_checkpoint_path)
        model_mlp.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)
        val_loss, val_acc = model_mlp.evaluate(x_valid,y_valid)


    # train generator is to be switched to set_retrieval mode  
    #########################################　削除部分　##########################################
    train_generator = data.trainDataGenerator(year=args.year, batch_size=args.batch_size, max_item_num=args.max_item_num)
    x_valid, x_size_valid, y_valid = train_generator.data_generation_val()
    #########################################　削除部分　##########################################

else:
    # train generator is to be switched to set_retrieval mode  
    #########################################　削除部分　##########################################
    train_generator = data.trainDataGenerator(year=args.year, batch_size=args.batch_size, max_item_num=args.max_item_num)
    x_valid, x_size_valid, y_valid = train_generator.data_generation_val()
    #########################################　削除部分　##########################################



#---------------------------------------------------------------------------------------


# # SetMatchingModelの定義と学習
set_matching_model = models.SetMatchingModel(
    isCNN=False,                     # CNNを使用するかどうか
    is_TrainableMLP=args.pretrained_mlp, # MLPを学習するかどうか（デフォルト: False）
    is_set_norm=args.is_set_norm,    # Set Normalizationの有無
    is_cross_norm=args.is_cross_norm,# Cross Normalizationの有無
    is_final_linear=True,            # 最終的な線形層を使用するかどうか（デフォルト: True）
    num_layers=args.num_layers,      # エンコーダ・デコーダのレイヤー数
    num_heads=args.num_heads,        # Attentionのヘッド数
    mode='setRepVec_pivot',          # モード（デフォルト: 'setRepVec_pivot'）
    baseChn=64,            # ベースのチャンネル数
    baseMlp=args.mlp_projection_dim * 4,              # ベースのMLPのチャンネル数
    rep_vec_num=1,                   # 代表ベクトルの数
    seed_init = seed_vectors,                   # シードの初期化
    cnn_class_num=2,                 # CNNの分類クラス数（デフォルト: 2）
    max_channel_ratio=2,             # チャンネル倍率
    is_neg_down_sample=args.is_neg_down_sample # ネガティブサンプルのダウンサンプリングを使用するかどうか
)

# 学習
SetMatching_model_path = f"{experimentPath}_TrainScore"
cp_callback = tf.keras.callbacks.ModelCheckpoint(SetMatching_model_path, monitor='val_Set_accuracy', save_weights_only=True, mode='max', save_best_only=True, save_freq='epoch', verbose=1)
cp_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_Set_accuracy', patience=args.patience, mode='max', min_delta=0.001, verbose=1)

set_matching_model.compile(optimizer='adam', loss=util.Set_hinge_loss, metrics=util.Set_accuracy)
set_matching_model.fit(train_generator, epochs=args.epochs, validation_data=((x_valid, x_size_valid), y_valid), callbacks=[cp_callback, cp_earlystopping])

# 事前学習モデルの保存
set_matching_model.save_weights('experiment_TrainScore/set_matching_weights.ckpt')
#---------------------------------------------------------------------------------------



# set-matching network
if args.model == 'VLAD':
    model = models.VLAD(num_features=args.baseChn, k_centers=args.rep_vec_num, batch_size=args.batch_size, seed_vectors = seed_vectors)
elif args.model == 'SMN':
    model = models.SMN(isCNN=False, is_TrainableMLP=args.pretrained_mlp, is_set_norm=args.is_set_norm, is_cross_norm=args.is_cross_norm, num_layers=args.num_layers, \
                           num_heads=args.num_heads, baseChn=args.baseChn, baseMlp = baseMLPChn, mode=args.mode, calc_set_sim=args.calc_set_sim, rep_vec_num=args.rep_vec_num, seed_init = seed_vectors, is_neg_down_sample=args.is_neg_down_sample, use_Cvec = args.use_Cvec, is_Cvec_linear=args.is_Cvec_linear)
elif args.model == 'random':
    model = models.random()
elif args.model == 'CNN':
    model = models.CNN()

# model.build([(100, 5, 4096), (100,)])
# model.summary()

if not os.path.exists(main_model_path):
    os.makedirs(main_model_path)
    os.makedirs(os.path.join(main_model_path, 'model'))
    os.makedirs(os.path.join(main_model_path, 'result'))

main_checkpoint_path = os.path.join(main_model_path,"model/cp.ckpt")
main_result_path = os.path.join(main_model_path,"result/result.pkl")

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

cp_callback = tf.keras.callbacks.ModelCheckpoint(main_checkpoint_path, monitor='val_Set_accuracy', save_weights_only=True, mode='max', save_best_only=True, save_freq='epoch', verbose=1)
cp_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_Set_accuracy', patience=args.patience, mode='max', min_delta=0.001, verbose=1)

#########################################　追加部分　##########################################

# train_generator = data.DataGenerator(year=args.year, batch_size=args.batch_size, max_item_num=args.max_item_num)

# train_data_set = train_generator.get_train_dataset()
# validation_data_set = train_generator.get_validation_dataset()
# train_data_set = train_data_set.prefetch(1)
# validation_data_set = validation_data_set.prefetch(1)

#########################################　追加部分　##########################################

# for batch in train_data_set.take(1):
#     x_batch, y_batch = batch

# Check if a checkpoint exists and load it if it does
# latest = tf.train.latest_checkpoint(main_checkpoint_dir)
# if latest:
#     print(f"Loading weights from {latest}")
#     model.load_weights(latest)
#     initial_epoch = int(latest.split('_')[-1])
# else:
#     initial_epoch = 0

#################################################################################################
# if not os.path.exists(main_result_path): # if mode is train とかに変える
if args.train:
    # setting training, loss, metric to model
    model.compile(optimizer="adam", loss=util.Set_hinge_loss, metrics=util.Set_accuracy, run_eagerly=True)
    
    # train_steps, validation_steps = train_generator.__len__()

    # execute training
    # history = model.fit(
    #     train_generator, 
    #     steps_per_epoch=train_steps, 
    #     epochs=args.epochs, 
    #     validation_data=validation_data_set, 
    #     validation_steps=validation_steps,
    #     shuffle=True, 
    #     callbacks=[cp_callback,cp_earlystopping],
    #     verbose=1
    #     ) # 吉田法
    
    # history = model.fit(x=(x_train, x_size_train), y=y_train, validation_data=((x_valid, x_size_valid), y_valid), epochs=args.epochs, shuffle=True, callbacks=[cp_callback, cp_earlystopping])
    history = model.fit(train_generator, epochs=args.epochs, validation_data=((x_valid, x_size_valid), y_valid), shuffle=True, callbacks=[cp_callback,cp_earlystopping]) #山園法
    # model.evaluate((x_valid, x_size_valid), y_valid)
    # loss, accuracy = model.evaluate((x_valid, x_size_valid), y_valid)
    # print(f"Validation loss: {loss}, Validation accuracy: {accuracy}")
    # accuracy and loss
    acc = history.history['Set_accuracy']
    val_acc = history.history['val_Set_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # plot loss & acc
    util.plotLossACC(main_model_path,loss,val_loss,acc,val_acc)

    # dump to pickle
    with open(main_result_path,'wb') as fp:
        pickle.dump(acc,fp)
        pickle.dump(val_acc,fp)
        pickle.dump(loss,fp)
        pickle.dump(val_loss,fp)

else:
    # load trained parameters
    print("load models")
    model.load_weights(main_checkpoint_path)

# モデルパラメータの保存
model.save_weights(main_checkpoint_path)

################################################################################################