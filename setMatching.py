import tensorflow as tf
import tensorflow_addons as tfa
import os
import matplotlib.pylab  as plt
import numpy as np
from models import layer_normalization, set_attention
import pdb


#----------------------------
# set matching network
class SMN(tf.keras.Model):
    def __init__(self, isCNN=True, is_set_norm=False, is_cross_norm=True, is_final_linear=True, num_conv_layers=3, num_layers=1, num_heads=2, mode='setRepVec_pivot', baseChn=32, rep_vec_num=1, cnn_class_num=2, max_channel_ratio=2, is_neg_down_sample=False):
        super(SMN, self).__init__()
        self.isCNN = isCNN
        self.num_conv_layers = num_conv_layers
        self.num_layers = num_layers
        self.mode = mode
        self.rep_vec_num = rep_vec_num
        self.baseChn = baseChn
        self.is_final_linear = is_final_linear
        self.is_neg_down_sample = is_neg_down_sample

        #---------------------
        # cnn
        self.CNN = []
        self.fc_cnn_proj = tf.keras.layers.Dense(baseChn*max_channel_ratio, activation=tfa.activations.gelu, use_bias=False, name='setmatching')
        #---------------------
        
        #---------------------
        # encoder
        self.set_emb = self.add_weight(name='set_emb',shape=(1,1,self.rep_vec_num,baseChn*max_channel_ratio),trainable=True)
        self.self_attentions = [set_attention(head_size=baseChn*max_channel_ratio, num_heads=num_heads, self_attention=True) for i in range(num_layers)]
        self.layer_norms_enc1 = [layer_normalization(size_d=baseChn*max_channel_ratio, is_set_norm=is_set_norm) for i in range(num_layers)]
        self.layer_norms_enc2 = [layer_normalization(size_d=baseChn*max_channel_ratio, is_set_norm=is_set_norm) for i in range(num_layers)]
        self.fcs_enc = [tf.keras.layers.Dense(baseChn*max_channel_ratio, activation=tfa.activations.gelu, use_bias=False, name='setmatching') for i in range(num_layers)]        
        #---------------------

        #---------------------
        # decoder
        self.cross_attentions = [set_attention(head_size=baseChn*max_channel_ratio, num_heads=num_heads) for i in range(num_layers)]
        self.layer_norms_dec1 = [layer_normalization(size_d=baseChn*max_channel_ratio, is_set_norm=is_set_norm, is_cross_norm=is_cross_norm) for i in range(num_layers)]
        self.layer_norms_dec2 = [layer_normalization(size_d=baseChn*max_channel_ratio, is_set_norm=is_set_norm, is_cross_norm=is_cross_norm) for i in range(num_layers)]
        self.fcs_dec = [tf.keras.layers.Dense(baseChn*max_channel_ratio, activation=tfa.activations.gelu, use_bias=False, name='setmatching') for i in range(num_layers)]
        #---------------------
     
        #---------------------
        # head network
        self.pma = set_attention(head_size=baseChn*max_channel_ratio, num_heads=num_heads)  # poolingMA
        self.fc_final1 = tf.keras.layers.Dense(baseChn, name='setmatching')
        self.fc_final2 = tf.keras.layers.Dense(1, activation='sigmoid', name='setmatching')
        self.fc_proj = tf.keras.layers.Dense(1, use_bias=False, name='projection')  # linear projection
        #---------------------

    # compute score of set-pair using dot product
    def dot_set_score(self, x):
        nSet_x, nSet_y, dim = x.shape
       
        score = tf.stack([[tf.tensordot(x[i,j],x[j,i],1) for i in range(nSet_x)] for j in range(nSet_y)])
        score = tf.expand_dims(score,-1)/tf.cast(dim,float)

        return score

    def call(self, x):
        x, x_size = x

        debug = {}
        shape = tf.shape(x)
        nSet = shape[0]
        nItemMax = shape[1]

        # CNN
        if self.isCNN:
            x, predCNN = self.CNN((x,x_size),training=False)
        else:
            x = self.fc_cnn_proj(x)
            predCNN = []
        
        # reshape (nSet*nItemMax, D) to (nSet, nItemMax, D)
        x = tf.reshape(x,[nSet, nItemMax, -1])

        # reshape (nSet, nItemMax, D) -> (nSet, nSet, nItemMax, D)
        x = tf.tile(tf.expand_dims(x,1),[1,nSet,1,1])

        debug['x_cnn'] = x

        # add_embedding
        x_orig = x
        if self.mode.find('setRepVec') > -1:
            set_emb_tile = tf.tile(self.set_emb, [nSet,nSet,1,1])
            x = tf.concat([set_emb_tile,x], axis=2)
            x_size += 1
        
        debug['x_encoder_layer_0'] = x

        x_2enc = x

        #---------------------
        # encoder (self-attention)
        for i in range(self.num_layers):

            if self.mode.find('setRepVec') > -1:
                self.self_attentions[i].rep_vec_num = self.rep_vec_num

            if self.mode == 'setRepVec_singlePMA': # single-PMA + single-PMA
                self.self_attentions[i].singlePMA = True

            z = self.layer_norms_enc1[i](x,x_size)

            # input: (nSet, nSet, nItemMax, D), output:(nSet, nSet, nItemMax, D)
            z = self.self_attentions[i](z,z)
            x += z

            z = self.layer_norms_enc2[i](x,x_size)
            z = self.fcs_enc[i](z)
            x += z

            debug[f'x_encoder_layer_{i+1}'] = x
        x_enc = x
        #---------------------

        #---------------------
        # decoder (cross-attention)
        debug[f'x_decoder_layer_0'] = x
        for i in range(self.num_layers):

            if self.mode.find('setRepVec') > -1:
                self.cross_attentions[i].rep_vec_num = self.rep_vec_num            

            if self.mode == 'setRepVec_pivot': # Bi-PMA + pivot-cross
                self.cross_attentions[i].pivot_cross = True

            elif self.mode == 'setRepVec_singlePMA':  # single-PMA + single-PMA
                self.cross_attentions[i].singlePMA = True

            z = self.layer_norms_dec1[i](x,x_size)

            # input: (nSet, nSet, nItemMax, D), output:(nSet, nSet, nItemMax, D)
            z = self.cross_attentions[i](z,z)
            x += z
    
            z = self.layer_norms_dec2[i](x,x_size)
            z = self.fcs_dec[i](z)
            x += z

            debug[f'x_decoder_layer_{i+1}'] = x
        x_dec = x
        #---------------------

        #---------------------
        # calculation of score
        if self.mode=='maxPooling':
            # zero-padding mask
            shape = tf.shape(x)
            mask = tf.tile(tf.reduce_sum(x,axis=-1,keepdims=1)!=0,[1,1,1,shape[-1]])

            x_inf = tf.where(mask,x,tf.ones_like(x)*-np.inf)
            x_rep = tf.reduce_max(x,axis=2)   #(nSet,nSet,nItemMax,D) -> (nSet,nSet,D)

            score = self.dot_set_score(x_rep)

        elif self.mode.find('setRepVec') > -1:    # representative vec            
            x_rep = x[:,:,:self.rep_vec_num,:] #(nSet,nSet,nItemMax+1,D) -> (nSet,nSet,D)
            shape = x_rep.shape
            x_rep = tf.reshape(x_rep,[shape[0],shape[1],-1])

            score = self.dot_set_score(x_rep) #(nSet,nSet,D) -> (nSet, nSet)
   
        elif self.mode=='poolingMA':  # pooling by multihead attention            
            # create Seed Vector
            set_emb_tile = tf.tile(self.set_emb, [nSet,nSet,1,1])
            
            # PMA
            x_pma = self.pma(set_emb_tile,x) #(nSet,nSet,rep_vec_num,D), (nSet,nSet,nItemMax,D) -> (nSet,nSet,rep_vec_num,D)
            x_rep = x_pma[:,:,0,:]

            # calculate score
            score = self.dot_set_score(x_rep) #(nSet,nSet,D) -> (nSet,nSet)
            if np.isnan(np.max(score)):
                pdb.set_trace()

        debug['score'] = score
        
        # linearly convert matching-score to class-score
        size_d = tf.shape(score)[2]

        if self.is_final_linear:
            predSMN = self.fc_final2(tf.reshape(score,[-1,size_d]))
        else:
            fc_final1 = self.fc_final1(tf.reshape(score,[-1,size_d]))
            predSMN = self.fc_final2(fc_final1)
        #---------------------

        return predCNN, predSMN, debug

    # convert class labels to cross-set label（if the class-labels are same, 1, otherwise 0)
    def cross_set_label(self, y):
        # rows of table
        y_rows = tf.tile(tf.expand_dims(y,-1),[1,tf.shape(y)[0]])
        # cols of table       
        y_cols = tf.tile(tf.transpose(tf.expand_dims(y,-1)),[tf.shape(y)[0],1])

        # if the class-labels are same, 1, otherwise 0
        labels = tf.cast(y_rows == y_cols, float)            
        return labels

    def toBinaryLabel(self,y):
        dNum = tf.shape(y)[0]
        y = tf.map_fn(fn=lambda x:0 if tf.less(x,0.5) else 1, elems=tf.reshape(y,-1))

        return tf.reshape(y,[dNum,-1])

    def neg_down_sampling(self, y_true, y_pred):
        # split to positive or negative data
        mask_pos = tf.not_equal(y_true,0)
        mask_neg = tf.not_equal(y_true,1)
        
        # number of pos and neg
        num_pos = tf.reduce_sum(tf.cast(mask_pos,tf.int32))
        num_neg = tf.reduce_sum(tf.cast(mask_neg,tf.int32))
        
        # split
        y_true_pos = tf.boolean_mask(y_true,mask_pos)
        y_pred_pos = tf.boolean_mask(y_pred,mask_pos)
        y_true_neg = tf.boolean_mask(y_true,mask_neg)
        y_pred_neg = tf.boolean_mask(y_pred,mask_neg)

        # select neg data
        thre = tf.cast(1.0-num_pos/num_neg,float)
        mask_neg_thre = tf.greater(tf.random.uniform([num_neg]),thre)
        y_true_neg = tf.boolean_mask(y_true_neg,mask_neg_thre)
        y_pred_neg = tf.boolean_mask(y_pred_neg,mask_neg_thre)

        # concat
        y_true = tf.concat([y_true_pos,y_true_neg],axis=0)
        y_pred = tf.concat([y_pred_pos,y_pred_neg],axis=0)

        return y_true, y_pred

    # train step
    def train_step(self,data):
        x, y_true = data
        x, x_size = x

        with tf.GradientTape() as tape:
            # predict
            predCNN, predSMN, debug = self((x, x_size), training=True)
            
            y_pred = predSMN

            # convert to cross-set label
            y_true = self.cross_set_label(y_true)
            y_true = tf.reshape(y_true,-1)

            # mask for the pair of same sets
            mask = tf.not_equal(tf.reshape(tf.linalg.diag(tf.ones(x.shape[0])),-1),1)
            y_true = tf.boolean_mask(y_true, mask)
            y_pred = tf.boolean_mask(y_pred, mask)

            # down sampling
            if self.is_neg_down_sample:
                y_true, y_pred = self.neg_down_sampling(y_true, y_pred)

            # loss
            loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)
     
        # train using gradients
        trainable_vars = self.trainable_variables

        # train parameters excepts for CNN
        trainable_vars = [v for v in trainable_vars if 'cnn' not in v.name]
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(
            (grad, var)
            for (grad, var) in zip(gradients, trainable_vars)
            if grad is not None)

        # update metrics
        self.compiled_metrics.update_state(y_true, y_pred)

        # return metrics as dictionary
        return {m.name: m.result() for m in self.metrics}

    # test step
    def test_step(self, data):
        x, y_true = data
        x , x_size = x

        # predict
        predCNN, predSMN, debug = self((x, x_size), training=False)
        y_pred = predSMN

        # convert to cross-set label
        y_true = self.cross_set_label(y_true)
        y_true = tf.reshape(y_true,-1)

        # mask for the pair of same sets
        mask = tf.not_equal(tf.reshape(tf.linalg.diag(tf.ones(x.shape[0])),-1),1)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)

        # down sampling
        if self.is_neg_down_sample:
            y_true, y_pred = self.neg_down_sampling(y_true, y_pred)

        # loss
        self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)

        # update metrics
        self.compiled_metrics.update_state(y_true, y_pred)

        # return metrics as dictionary
        return {m.name: m.result() for m in self.metrics}

    # predict step
    def predict_step(self,data):
        batch_data = data[0]
        x, x_size = batch_data
        
        # predict
        predCNN, predSMN, debug = self((x, x_size), training=False)

        return predCNN, predSMN, debug
#----------------------------

#----------------------------
# plot loss and accuracy
def plotLossACC(path,loss,val_loss,acc,val_acc):
    epochs = np.arange(len(acc))

    fig=plt.figure()
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    fig.add_subplot(1,2,1)
    plt.plot(epochs,acc,'bo-',label='training acc')
    plt.plot(epochs,val_acc,'b',label='validation acc')
    plt.title('acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.ylim(0,1)
    
    fig.add_subplot(1,2,2)
    plt.plot(epochs,loss,'bo-',label='training loss')
    plt.plot(epochs,val_loss,'b',label='validation loss')
    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(0,3)
    plt.legend()

    path = os.path.join(path,"result/loss_acc.png")
    plt.savefig(path)
#----------------------------

#----------------------------
# plot histogram
def plotHist(corr_pos,corr_neg, mode, fname=''):
    fig = plt.figure(figsize=(20,5))

    max_data_num = np.max([len(corr_neg[0]),len(corr_pos[0])])
    for layer_ind in range(len(corr_pos)):
        fig.add_subplot(1,len(corr_pos),layer_ind+1)
        plt.hist(corr_neg[layer_ind],label="mismatch",bins=np.arange(-1,1.1,0.1))
        plt.hist(corr_pos[layer_ind],alpha=0.5,label="match",bins=np.arange(-1,1.1,0.1))        
        if layer_ind == 0:
            plt.legend(fontsize=12)
        plt.xlim([-1.2,1.2])
        plt.ylim([0,max_data_num])
        plt.xticks(fontsize=12)
        
    plt.tight_layout()

    if len(fname):
        plt.savefig(fname)
    else:
        plt.show()
#----------------------------

#----------------------------
# function to compute CMC
def calc_cmcs(pred, true_grp, batch_size, qry_ind=0, glry_start_ind=1, top_n=1):

    # reshape predict and true for each batch
    pred_batchs = np.reshape(pred, [-1, batch_size, batch_size])
    true_grp_batchs = np.reshape(true_grp, [-1, batch_size])

    # extract predicted scores for query and compute true labels 
    pred_scores = pred_batchs[:,qry_ind,glry_start_ind:]

    # label
    true_labs = (true_grp_batchs == true_grp_batchs[:,[qry_ind]])[:,glry_start_ind:].astype(int)

    # shuffle pred and true
    np.random.seed(0)
    random_inds = random_inds = np.vstack([np.random.permutation(len(true_labs[0])) for i in range(len(true_labs))]) 
    pred_scores = np.vstack([pred_scores[i][random_inds[i]] for i in range(len(random_inds))])
    true_labs = np.vstack([true_labs[i][random_inds[i]] for i in range(len(random_inds))])

    # sort predicted scores and compute TP map (data x batch_size)
    pred_sort_inds = np.argsort(pred_scores,axis=1)[:,::-1]
    TP_map = np.take_along_axis(true_labs,pred_sort_inds,axis=1)

    cmcs = np.sum(np.cumsum(TP_map,axis=1),axis=0)/len(true_labs)

    return cmcs
#----------------------------
