#coding=utf-8
import os
import sys
sys.path.append(os.path.abspath('../my_lib'))
from my_evaluation import *
sys.path.append(os.path.abspath('../my_lib/neural_module'))
from learn_strategy import LrWarmUp
from transformer import TranEnc,TranDec,DualTranDec
from embedding import PosEnc
from my_loss import LabelSmoothSoftmaxCEV2,CriterionNet
from balanced_data_parallel import BalancedDataParallel
sys.path.append(os.path.abspath('../my_lib/neural_model'))
from seq_to_seq_model import TransSeq2Seq
from base_model import BaseNet
# sys.path.append('../my_ib/neural_model')

from config import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset,DataLoader
import random
import numpy as np
import os
import logging
import pickle
import math
import codecs

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Datasetx(Dataset):
    '''
    文本对数据集对象（根据具体数据再修改）
    '''
    def __init__(self,ins,outs=None,in_max_len=None,out_max_len=None,out_begin_idx=1,out_end_idx=2):
        self.len=len(ins)  #样本个数
        self.in_max_len=in_max_len
        self.out_max_len=out_max_len
        self.out_begin_idx=out_begin_idx
        self.out_end_idx=out_end_idx
        if in_max_len is None:
            self.in_max_len = max([len(seqs[0]) for seqs in ins])   #每个输入有多类特征
        if out_max_len is None and outs is not None:
            self.out_max_len=max([len(seq) for seq in outs]) #每个输出只是一个序列
        self.ins=ins
        self.outs=outs
    def __getitem__(self, index):
        tru_feature=[seq[:self.in_max_len] for seq in self.ins[index]] #先做截断
        pad_feature = [np.lib.pad(seq, (0, self.in_max_len - len(seq)),
                                        'constant', constant_values=(0, 0)) for seq in tru_feature]  # padding
        # tru_out_inputs=[]
        if self.outs is None:
            pad_out_input=np.zeros((self.out_max_len+1,),dtype=np.long)   #decoder端的输入
            pad_out_input[0]=self.out_begin_idx
            return torch.tensor(pad_feature),\
                   torch.tensor(pad_out_input).long()
        else:
            tru_out = self.outs[index][:self.out_max_len]  # 先做截断
            pad_out_input=np.lib.pad(tru_out,(1,self.out_max_len-len(tru_out)),'constant',constant_values=(self.out_begin_idx, 0))
            tru_out=np.lib.pad(tru_out, (0,1),'constant', constant_values=(0, self.out_end_idx))  # padding
            pad_out= np.lib.pad(tru_out, (0, self.out_max_len+1 - len(tru_out)),
                                      'constant', constant_values=(0, 0))  # padding
            # pad_out_input=np.lib.pad(pad_out[:-1],(1,0),'constant',constant_values=(self.out_begin_idx, 0))
            return torch.tensor(pad_feature), \
                   torch.tensor(pad_out_input).long(), \
                   torch.tensor(pad_out).long()

    def __len__(self):
        return self.len

class Encoder(nn.Module):
    def __init__(self,
                 in_max_len,
                 in_node_voc_size,
                 in_ast_pos_voc_size,
                 in_edge_voc_size,
                 enc_node_init_emb=None,
                 enc_node_emb_freeze=True,
                 enc_edge_init_emb=None,
                 enc_edge_emb_freeze=True,
                 enc_emb_dim=300,
                 enc_att_layer_num=6,
                 enc_head_num=10,
                 enc_head_dim=None,
                 enc_ff_hid_dim=2048,
                 enc_drop_rate=0.,
                 ):
        super().__init__()
        self.in_max_len = in_max_len
        self.enc_emb_dim = enc_emb_dim
        if enc_node_init_emb is None:
            self.node_embedding = nn.Embedding(in_node_voc_size, enc_emb_dim, padding_idx=0)
            # nn.init.xavier_uniform_(self.node_embedding.weight[1:, :])  # nn.init.xavier_uniform_
            # self.position_encoder.weight.data[0, :] = 0  # 上面初始化后padding0被黑了，靠
        else:
            # print(init_emb.size())
            # assert init_emb.shape==(vocab_size,enc_emb_dim)
            enc_node_init_emb = torch.tensor(enc_node_init_emb, dtype=torch.float32)
            self.node_embedding = nn.Embedding.from_pretrained(enc_node_init_emb, freeze=enc_node_emb_freeze, padding_idx=0)
        self.node_pos_embedding = nn.Embedding(in_ast_pos_voc_size, enc_emb_dim, padding_idx=0)
        if enc_edge_init_emb is None:
            self.edge_embedding = nn.Embedding(in_edge_voc_size, enc_emb_dim, padding_idx=0)
            # nn.init.xavier_uniform_(self.edge_embedding.weight[1:, :])  # nn.init.xavier_uniform_
            # self.position_encoder.weight.data[0, :] = 0  # 上面初始化后padding0被黑了，靠
        else:
            # print(init_emb.size())
            # assert init_emb.shape==(vocab_size,enc_edge_emb_dim)
            enc_edge_init_emb = torch.tensor(enc_edge_init_emb, dtype=torch.float32)
            self.edge_embedding = nn.Embedding.from_pretrained(enc_edge_init_emb, freeze=enc_edge_emb_freeze, padding_idx=0)
        self.edge_pos_embedding=nn.Embedding(in_ast_pos_voc_size,enc_emb_dim,padding_idx=0)
        self.node_enc=TranEnc(query_dim=enc_emb_dim,
                             head_num=enc_head_num,
                             head_dim=enc_head_dim,
                             layer_num=enc_att_layer_num,
                              ff_hid_dim=enc_ff_hid_dim,
                             drop_rate=enc_drop_rate)
        self.edge_enc = TranEnc(query_dim=enc_emb_dim,
                                head_num=enc_head_num,
                                head_dim=enc_head_dim,
                                layer_num=enc_att_layer_num,
                                ff_hid_dim=enc_ff_hid_dim,
                                drop_rate=enc_drop_rate)

        # self.linear = nn.Linear(enc_emb_dim, enc_emb_dim)
        self.node_layer_norm = nn.LayerNorm(enc_emb_dim, elementwise_affine=True)
        self.edge_layer_norm = nn.LayerNorm(enc_emb_dim, elementwise_affine=True)
        self.enc_dropout = nn.Dropout(p=enc_drop_rate)
        # if self.out_dim>1:
        #     self.softmax=nn.Softmax(dim=-1)


    def forward(self, x):
        '''

        :param x: [B,5,L1]
        :return:
        '''
        #encoding:
        node_emb=self.node_embedding(x[:,0,:])*np.sqrt(self.enc_emb_dim)   #(B,L_x,D)
        node_pos_emb=self.node_pos_embedding(x[:,1,:])  #(B,L_x,D)
        node_encoder=self.enc_dropout(node_emb.add(node_pos_emb))
        node_encoder=self.node_layer_norm(node_encoder)
        # node_encoder=self.node_layer_norm(self.enc_dropout(self.node_embedding(x[:,0,:])))
        node_mask = x[:,0,:].abs().sign()  # (B,L)
        node_encoder=self.node_enc(query=node_encoder,query_mask=node_mask)    #(B,L_x,D)

        edge_emb = self.edge_embedding(x[:, 2, :])*np.sqrt(self.enc_emb_dim)  # (B,L_x,D)
        edge_pos_emb = self.edge_pos_embedding(x[:, 3, :])  # (B,L_x,D)
        edge_encoder =self.enc_dropout(edge_emb.add(edge_pos_emb))
        edge_encoder = self.edge_layer_norm(edge_encoder)
        # edge_encoder = self.edge_layer_norm(self.enc_dropout(self.edge_embedding(x[:, 4, :])))
        edge_mask = x[:, 2, :].abs().sign()  # (B,L)
        edge_encoder = self.edge_enc(query=edge_encoder, query_mask=edge_mask)  # (B,L_x,D)

        return torch.cat([node_encoder.unsqueeze(-1),edge_encoder.unsqueeze(-1)],dim=-1)  #(B,out_dim,L_y)

class Decoder(nn.Module):
    def __init__(self,
                 out_max_len,
                 out_word_voc_size,
                 out_dim,
                 enc_out_dim=300,
                 dec_word_emb_dim=300,
                 dec_word_init_emb=None,
                 dec_word_emb_freeze=True,
                 dec_att_layer_num=6,
                 dec_head_num=10,
                 dec_head_dim=None,
                 dec_ff_hid_dim=2048,
                 dec_drop_rate=0.
                 ):
        super().__init__()
        self.out_max_len=out_max_len+1
        self.dec_word_emb_dim=dec_word_emb_dim
        if dec_word_init_emb is None:
            self.word_embedding = nn.Embedding(out_word_voc_size, dec_word_emb_dim, padding_idx=0)
            # nn.init.xavier_uniform_(self.word_embedding.weight[1:, :])  # nn.init.xavier_uniform_
        else:
            dec_word_init_emb = torch.tensor(dec_word_init_emb, dtype=torch.float32)
            self.word_embedding = nn.Embedding.from_pretrained(dec_word_init_emb, freeze=dec_word_emb_freeze, padding_idx=0)

        self.pos_decoding = PosEnc(max_len=self.out_max_len, emb_dim=self.dec_word_emb_dim, train=True,pad=True)

        self.decoding=DualTranDec(query_dim=dec_word_emb_dim,
                              key_dim=enc_out_dim,
                              head_num=dec_head_num,
                              head_dim=dec_head_dim,
                              layer_num=dec_att_layer_num,
                              ff_hid_dim=dec_ff_hid_dim,
                              drop_rate=dec_drop_rate, )

        self.layer_norm=nn.LayerNorm(dec_word_emb_dim, elementwise_affine=True)
        self.out_fc = nn.Sequential(
            nn.Linear(dec_word_emb_dim, out_dim),
            # nn.LeakyReLU(),
            # nn.Dropout(dec_drop_rate),
            # nn.Linear(128, out_dim)
        )
        self.dec_dropout = nn.Dropout(p=dec_drop_rate)
        # if self.out_dim>1:
        #     self.softmax=nn.Softmax(dim=-1)


    def forward(self, y,x):
        '''

        :param x: [B,L_x,D]
        :param y: [B,L_y]
        :return:
        '''
        node_encoder=x[:,:,:,0]
        edge_encoder=x[:,:,:,1]
        node_mask = node_encoder.abs().sum(-1).sign()  # (B,L_x)
        edge_mask = edge_encoder.abs().sum(-1).sign()  # (B,L_x)
        #decoding:
        y_decoder=self.word_embedding(y)*np.sqrt(self.dec_word_emb_dim)   #(B,L_y,D)
        y_pos_emb=self.pos_decoding(y)    #(B,L_y,D)
        y_decoder=self.dec_dropout(y_decoder.add(y_pos_emb))
        y_decoder = self.layer_norm(y_decoder)
        # y_decoder = self.layer_norm(self.dec_dropout(self.word_embedding(y)))
        y_mask=y.abs().sign()
        # print(node_encoder.size(),y_decoder.size())
        y_decoder=self.decoding(query=y_decoder,
                                key1=node_encoder,key2=edge_encoder,
                                query_mask=y_mask,
                                key_mask1=node_mask,key_mask2=edge_mask
                                )   #(B,L_y,D)

        #output:
        outputs=self.out_fc(y_decoder)     #(B,L_y,out_dim)
        # else:
        #     outputs=self.softmax(outputs) #pytorch大坑：使用nn.CrossEntropyLoss时，千万不要在输出前做nn.Softmax
        return outputs.transpose(1, 2)  #(B,out_dim,L_y)

class TransNet(BaseNet):
    def __init__(self,
                 in_max_len,
                 out_max_len,
                 in_node_voc_size,
                 in_ast_pos_voc_size,
                 in_edge_voc_size,
                 # in_edge_pos_voc_size,
                 out_word_voc_size,
                 out_dim,
                 enc_emb_dim=300,
                 dec_word_emb_dim=300,
                 enc_node_init_emb=None,
                 enc_edge_init_emb=None,
                 dec_word_init_emb=None,
                 enc_node_emb_freeze=True,
                 enc_edge_emb_freeze=True,
                 dec_word_emb_freeze=True,
                 enc_att_layer_num=6,
                 dec_att_layer_num=6,
                 enc_head_num=10,
                 dec_head_num=10,
                 enc_head_dim=None,
                 dec_head_dim=None,
                 enc_ff_hid_dim=2048,
                 dec_ff_hid_dim=2048,
                 enc_drop_rate=0.,
                 dec_drop_rate=0.
                 ):
        super().__init__()
        self.init_params = locals()
        # self.in_max_len = in_max_len
        # self.out_max_len=out_max_len+1
        self.encoder=Encoder(in_max_len=in_max_len,
                             in_node_voc_size=in_node_voc_size,
                             in_ast_pos_voc_size=in_ast_pos_voc_size,
                             in_edge_voc_size=in_edge_voc_size,
                             # in_edge_pos_voc_size=in_edge_pos_voc_size,
                             enc_emb_dim=enc_emb_dim,
                             enc_node_init_emb=enc_node_init_emb,
                             enc_node_emb_freeze=enc_node_emb_freeze,
                             enc_edge_init_emb=enc_edge_init_emb,
                             enc_edge_emb_freeze=enc_edge_emb_freeze,
                             enc_att_layer_num=enc_att_layer_num,
                             enc_head_num=enc_head_num,
                             enc_head_dim=enc_head_dim,
                             enc_ff_hid_dim=enc_ff_hid_dim,
                             enc_drop_rate=enc_drop_rate)
        self.decoder=Decoder(out_max_len=out_max_len,
                             out_word_voc_size=out_word_voc_size,
                             out_dim=out_dim,
                             enc_out_dim=enc_emb_dim,
                             dec_word_emb_dim=dec_word_emb_dim,
                             dec_word_init_emb=dec_word_init_emb,
                             dec_word_emb_freeze=dec_word_emb_freeze,
                             dec_att_layer_num=dec_att_layer_num,
                             dec_head_num=dec_head_num,
                             dec_head_dim=dec_head_dim,
                             dec_ff_hid_dim=dec_ff_hid_dim,
                             dec_drop_rate=dec_drop_rate)


    def forward(self, x,y):
        '''

        :param x: [B,5,L1]
        :param y: [B,L2]
        :return:
        '''
        #encoding:
        x_encoder=self.encoder(x)
        #decoding:
        y_decoder=self.decoder(y,x_encoder)
        return y_decoder  #(B,out_dim,L_y)


class TModel(TransSeq2Seq):
    def __init__(self,
                 model_dir,
                 model_name='Transformer_based_model',
                 model_id=None,
                 enc_emb_dim=150,
                 dec_word_emb_dim=150,
                 enc_node_emb_path=None,
                 enc_edge_emb_path=None,
                 dec_word_emb_path=None,
                 enc_node_emb_freeze=False,
                 enc_edge_emb_freeze=False,
                 dec_word_emb_freeze=False,
                 enc_att_layer_num=6,
                 dec_att_layer_num=6,
                 enc_drop_rate=0.3,
                 dec_drop_rate=0.3,
                 enc_head_num=8,
                 enc_head_dim=None,
                 dec_head_num=8,
                 dec_head_dim=None,
                 enc_ff_hid_dim=2048,
                 dec_ff_hid_dim=2048,
                 batch_size=32,
                 big_epochs=20,
                 regular_rate=1e-5,
                 lr_base=0.001,
                 lr_decay=0.9,
                 min_lr_rate=0.01,
                 warm_big_epochs=2,
                 Net=TransNet,
                 Dataset=Datasetx
                 ):
        '''
        构造函数
        :param model_dir: 模型存放目录
        :param model_name: 模型名称
        :param model_id: 模型id
        :param max_class_num: 一个apk里最大类数量
        :param enc_emb_dim: 词向量维度
        :param head_num: header的数量
        :param att_layer_num: 每次transformer的模块数量
        :param drop_rate: dropout rate
        :param batch_size: 批处理数据量
        :param big_epochs: 总体训练迭代次数（对整体数据集遍历了几次）
        :param regular_rate: 正则化比率
        :param lr_base: 初始学速率
        :param lr_decay: 学习速率衰减率
        :param staircase: 学习率是否梯度衰减，即是不是遍历完所有数据再衰减
        '''
        logging.info('Construct %s'%model_name)
        super().__init__(model_name=model_name,
                         model_dir=model_dir,
                         model_id=model_id)
        self.init_params = locals()
        # self.Dataset = Datasetx
        self.enc_emb_dim = enc_emb_dim
        self.dec_word_emb_dim = dec_word_emb_dim
        self.enc_node_emb_path = enc_node_emb_path
        self.enc_edge_emb_path = enc_edge_emb_path
        self.dec_word_emb_path = dec_word_emb_path
        self.enc_node_emb_freeze = enc_node_emb_freeze
        self.enc_edge_emb_freeze = enc_edge_emb_freeze
        self.dec_word_emb_freeze = dec_word_emb_freeze
        self.enc_att_layer_num = enc_att_layer_num
        self.dec_att_layer_num = dec_att_layer_num
        self.enc_drop_rate = enc_drop_rate
        self.dec_drop_rate = dec_drop_rate
        self.enc_head_num = enc_head_num
        self.enc_head_dim = enc_head_dim
        self.dec_head_num = dec_head_num
        self.dec_head_dim = dec_head_dim
        self.enc_ff_hid_dim=enc_ff_hid_dim
        self.dec_ff_hid_dim=dec_ff_hid_dim
        self.batch_size = batch_size
        self.big_epochs = big_epochs
        self.regular_rate = regular_rate
        self.lr_base = lr_base
        self.lr_decay = lr_decay
        self.min_lr_rate = min_lr_rate
        self.warm_big_epochs = warm_big_epochs
        self.Net = Net
        self.Dataset = Dataset


    def fit(self,
            train_ins,
            train_outs,
            out_word_w2i,  # token的映射字典
            out_word_i2w,  # token的反映射字典
            valid_ins=None,
            valid_outs=None,
            in_max_len=None,
            out_max_len=None,
            in_node_voc_size=None,
            in_edge_voc_size=None,
            in_ast_pos_voc_size=None,
            out_word_voc_size=None,
            out_dim=None,
            train_metrics=[get_overall_accuracy],
            valid_metric=get_overall_accuracy,
            verbose=0,
            train=True
            ):
        # print(self.__dict__)
        '''
        训练模型接口
        :param train_ins: 特征集，结构为[[文本ndarray,文本ndarray],[文本ndarray,文本ndarray],...],双层list+ndarray文本对
        :param train_outs: 输出标记
        :param use_tensorboard: 是否使用tensorboard
        :param verbose: 训练时显示日志信息，为0表示不输出日志，为1每个batch都输出，为2表示每遍历完一次所有数据输出一次
        :param train_metrics: 一个列表的用于训练时对训练数据的评价函数
        :param valid_metric: 用于训练时对验证数据的评价函数，如果为None，则用loss
        :return:
        '''
        logging.info('Train %s'%self.model_name)
        self.train_metrics = train_metrics
        self.valid_metric=valid_metric
        # torch.autograd.set_detect_anomaly(True)
        #计算最大输入输出序列长度
        if in_max_len is None:
            self.in_max_len = max([len(seqs[0]) for seqs in train_ins])  #最大输入长度
        else:
            self.in_max_len=in_max_len
        if out_max_len is None:
            self.out_max_len = max([len(seq) for seq in train_outs])  #最大输出长度
        else:
            self.out_max_len=out_max_len
        # self.out_max_len=50  #最大输出长度
        #计算词库大小
        if in_node_voc_size is None:
            self.in_node_voc_size=max([np.max(seqs[0,:]) for seqs in train_ins])+1 #词表大小，词表从0开始，因此要最大序号+1才行(unknown标记也算进去了）
        else:
            self.in_node_voc_size=in_node_voc_size
        # self.in_ast_pos_voc_size=max([np.max(seqs[1,:]) for seqs in train_ins])+1
        if in_edge_voc_size is None:
            self.in_edge_voc_size = max([np.max(seqs[2, :]) for seqs in train_ins]) + 1  # 词表大小，词表从0开始，因此要最大序号+1才行(unknown标记也算进去了）
        else:
            self.in_edge_voc_size=in_edge_voc_size
        if in_ast_pos_voc_size is None:
            self.in_ast_pos_voc_size = max([np.max(seqs[[1,3], :]) for seqs in train_ins]) + 1
        else:
            self.in_ast_pos_voc_size=in_ast_pos_voc_size
        if out_word_voc_size is None:
            self.out_word_voc_size=max(np.max(seq) for seq in train_outs)+3  #因为又一个begin id和一个end_id
        else:
            self.out_word_voc_size=out_word_voc_size
        if out_dim is None:
            self.out_dim=max([np.max(seq) for seq in train_outs])+2 #输出维度大小，也是输出词表大小，0为<PAD>，1为<UNK>,max+
        else:
            self.out_dim=out_dim
        # print(self.out_max_len)
        # print(self.in_max_len)

        self.out_word_w2i=out_word_w2i
        self.out_word_i2w=out_word_i2w    #字典
        self.out_i2w=out_word_i2w
        # self.token_i2w=self.out_word_i2w

        # print(self.sort_unique_outs)
        enc_node_emb_weight=None
        if self.enc_node_emb_path is not None: # 如果加载预训练词向量
            enc_node_emb_weight=np.load(self.enc_node_emb_path)
            # print(enc_node_emb_weight[-1,:])
            # print(np.linalg.norm(enc_node_emb_weight, axis=1, keepdims=True)[0,:])
            # enc_node_emb_weight[1:,:]/=np.linalg.norm(enc_node_emb_weight[1:,:], axis=1, keepdims=True)    #归一化,第1行都是0不能参加运算
            # print(enc_node_emb_weight[-1, :])
            self.in_node_voc_size = enc_node_emb_weight.shape[0]
            # print(enc_node_emb_weight[2,:])
            # print(enc_node_emb_weight.shape)
            # print(self.sort_unique_outs)
        enc_edge_emb_weight = None
        if self.enc_edge_emb_path is not None: # 如果加载预训练词向量
            enc_edge_emb_weight=np.load(self.enc_edge_emb_path)
            # print(enc_edge_emb_weight[-1,:])
            # print(np.linalg.norm(enc_edge_emb_weight, axis=1, keepdims=True)[0,:])
            # enc_edge_emb_weight[1:,:]/=np.linalg.norm(enc_edge_emb_weight[1:,:], axis=1, keepdims=True)    #归一化,第1行都是0不能参加运算
            # print(enc_edge_emb_weight[-1, :])
            self.in_edge_voc_size = enc_edge_emb_weight.shape[0]
            # print(enc_edge_emb_weight[2,:])
            # print(enc_edge_emb_weight.shape)
            # print(self.sort_unique_outs)
        dec_word_emb_weight = None
        if self.dec_word_emb_path is not None:  # 如果加载预训练词向量
            dec_word_emb_weight = np.load(self.dec_word_emb_path)
            # print(dec_word_emb_weight[-1,:])
            # print(np.linalg.norm(dec_word_emb_weight, axis=1, keepdims=True)[0,:])
            # dec_word_emb_weight[1:,:]/=np.linalg.norm(dec_word_emb_weight[1:,:], axis=1, keepdims=True)    #归一化,第1行都是0不能参加运算
            # print(dec_word_emb_weight[-1, :])
            self.out_word_voc_size = dec_word_emb_weight.shape[0]
            # print(dec_word_emb_weight[2,:])
            # print(dec_word_emb_weight.shape)
        # print()
        # self.in_max_len = 38
        net = self.Net(in_max_len=self.in_max_len,
                       out_max_len=self.out_max_len,
                       in_node_voc_size=self.in_node_voc_size,
                       in_ast_pos_voc_size=self.in_ast_pos_voc_size,
                       in_edge_voc_size=self.in_edge_voc_size,
                       # in_edge_pos_voc_size=self.in_edge_pos_voc_size,
                       out_word_voc_size=self.out_word_voc_size,
                       out_dim=self.out_dim,
                       enc_emb_dim=self.enc_emb_dim,
                       dec_word_emb_dim=self.dec_word_emb_dim,
                       enc_node_init_emb=enc_node_emb_weight,
                       enc_edge_init_emb=enc_edge_emb_weight,
                       dec_word_init_emb=dec_word_emb_weight,
                       enc_node_emb_freeze=self.enc_node_emb_freeze,
                       enc_edge_emb_freeze=self.enc_edge_emb_freeze,
                       dec_word_emb_freeze=self.dec_word_emb_freeze,
                       enc_att_layer_num=self.enc_att_layer_num,
                       dec_att_layer_num=self.dec_att_layer_num,
                       enc_head_num=self.enc_head_num,
                       enc_head_dim=self.enc_head_dim,
                       dec_head_num=self.dec_head_num,
                       dec_head_dim=self.dec_head_dim,
                       enc_ff_hid_dim=self.enc_ff_hid_dim,
                       dec_ff_hid_dim=self.dec_ff_hid_dim,
                       enc_drop_rate=self.enc_drop_rate,
                       dec_drop_rate=self.dec_drop_rate)

        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') #选择GPU优先
        # self.net = nn.DataParallel(net.to(device))  # 并行使用多GPU
        self.net = BalancedDataParallel(0, net.to(device), dim=0)  # 并行使用多GPU
        # self.net.to(device) #数据转移到设备

        self.net.train()    #设置网络为训练模式

        # for p in self.net.parameters(): #初始化非embedding层的参数
        #     if p.size(0) != self.vocab_size:
            # nn.init.normal_(p, 0.0, 0.01)

        # self.best_net=None
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.lr_base,
                                    weight_decay=self.regular_rate)
        if self.enc_node_emb_path is not None \
            and self.enc_edge_emb_path is not None \
                and self.dec_word_emb_path is not None:  # 如果加载预训练词向量
            node_emb_param = [x for x in self.net.parameters() if x.requires_grad and x.size(0) == self.in_node_voc_size]
            edge_emb_param = [x for x in self.net.parameters() if x.requires_grad and x.size(0) == self.in_edge_voc_size]
            word_emb_param = [x for x in self.net.parameters() if x.requires_grad and x.size(0) == self.out_word_voc_size]
            ex_param = [x for x in self.net.parameters() if x.requires_grad
                        and x.size(0) != self.in_node_voc_size
                        and x.size(0) != self.out_edge_voc_size
                        and x.size(0) != self.out_word_voc_size]
            optim_cfg = [{'params': node_emb_param, 'lr': self.lr_base},
                         {'params': edge_emb_param, 'lr': self.lr_base},
                         {'params': word_emb_param, 'lr': self.lr_base},
                         {'params': ex_param, 'lr': self.lr_base, 'weight_decay': self.regular_rate}, ]
            self.optimizer = optim.Adam(optim_cfg)
            # freeze_emb_param_ids=list(map(id,self.net.module.word_embedding.parameters()))
            # output_param_ids=list(map(id,self.net.module.out_fc.parameters()))
            # ex_param=filter(lambda p: id(p) not in freeze_emb_param_ids+output_param_ids,self.net.module.parameters())
            # optim_cfg = [{'params': self.net.module.word_embedding.parameters(), 'lr': self.learning_rate_base * 0.1},
            #              {'params': self.net.module.out_fc.parameters(), 'lr': self.learning_rate_base * 0.1},
            #              {'params': ex_param, 'lr': self.learning_rate_base, 'weight_decay': self.regular_rate}, ]
        # self.optimizer=optim.SGD(self.net.parameters(),lr=self.lr_base,weight_decay=self.regular_rate,momentum=0.9)

        # self.scheduler=lr_self.scheduler.ExponentialLR(self.optimizer,gamma=self.lr_decay)

        # self.scheduler=lr_self.scheduler.ReduceLROnPlateau(self.optimizer)

        # min_learning_rate=self.lr_base*0.1
        # self.scheduler=lr_self.scheduler.ReduceLROnPlateau(self.optimizer,
        #                                          mode='min',
        #                                          factor=self.lr_decay,
        #                                          patience=1,
        #                                          verbose=False,
        #                                          threshold=1e-4,
        #                                          threshold_mode='rel',
        #                                          cooldown=0,
        #                                          min_lr=min_learning_rate,
        #                                          eps=10e-8)

        # self.criterion = nn.CrossEntropyLoss(reduction='mean',ignore_index=0)
        self.criterion = LabelSmoothSoftmaxCEV2(reduction='mean',ignore_index=0,label_smooth=0.1)
        # self.criterion = nn.DataParallel(CriterionNet(criterion).to(device))
        # train_ins=sorted(train_ins,key=lambda x: len(x))  #根据文本长度对数据集排序

        #数据加载器
        self.out_begin_idx=self.out_word_w2i[OUT_BEGIN_TOKEN]
        self.out_end_idx=self.out_word_w2i[OUT_END_TOKEN]
        train_set = self.Dataset(train_ins,
                                 train_outs,
                                 in_max_len=self.in_max_len,
                                 out_max_len=self.out_max_len,
                                 out_begin_idx=self.out_begin_idx,
                                 out_end_idx=self.out_end_idx)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=8)

        # self.scheduler = lr_self.scheduler.CosineAnnealingLR(self.optimizer,
        #                                            T_max=train_loader.__len__(),
        #                                            eta_min=0,
        #                                            last_epoch=-1)
        # print(train_loader.__len__())
        if self.warm_big_epochs is None:
            self.warm_big_epochs= max(self.big_epochs // 10, 2)
        self.scheduler = LrWarmUp(self.optimizer,
                             min_rate=self.min_lr_rate,
                             lr_decay=self.lr_decay,
                             warm_steps=self.warm_big_epochs * len(train_loader),
                             # max(self.big_epochs//10,2)*train_loader.__len__()
                             reduce_steps=len(train_loader))  # 预热次数 train_loader.__len__()

        # with torch.autograd.set_detect_anomaly(True)
        # self.net.load_state_dict(torch.load(os.path.join(self.model_dir, self.model_name+'_best_net.net')))
        if train:
            for i in range(self.big_epochs):
                logging.info('---------Train big epoch %d/%d'%(i+1,self.big_epochs))
                for j, (batch_features,batch_out_inputs, batch_outs) in enumerate(train_loader):
                    #数据加入device
                    # if i==0 and j==0:
                    #     print(batch_out_inputs[0,:],batch_outs[0,:])
                    batch_features=batch_features.to(device)
                    batch_out_inputs=batch_out_inputs.to(device)
                    batch_outs=batch_outs.to(device)
                    # print(batch_features.size())

                    pred_outs=self.net(batch_features,batch_out_inputs)
                    del batch_features
                    loss=self.criterion(pred_outs,batch_outs)
                    self.optimizer.zero_grad()
                    loss.backward()
                    # clip_grad_norm(self.net.parameters(),1e-4)  #减弱梯度爆炸
                    self.optimizer.step()

                    self.scheduler.step()
                    # logging.info('The learning rates of first and last parameter group:{}, {}'.
                    #              format(self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[-1]['lr']))
                    # print(self.scheduler.get_lr()[0])
                    if i>(self.big_epochs-6):
                        self._log_fit_eval(loss=loss,
                                           big_step=i+1,
                                           batch_step=j+1,
                                           big_epochs=self.big_epochs,
                                           batch_epochs=len(train_loader),
                                           pred_outs=pred_outs,
                                           true_outs=batch_outs,
                                           seq_mode='BLEU',
                                           verbose=verbose,
                                           )
                    del pred_outs
                    del batch_outs

                    # if j==len(train_loader)//2:  # 根据验证集loss选择best_net
                    #     self._do_validation(valid_ins=valid_ins,
                    #                         valid_outs=valid_outs,
                    #                         increase_better=True,
                    #                         last=False)
                if i>=(self.big_epochs-6):
                    self._do_validation(valid_ins=valid_ins,
                                        valid_outs=valid_outs,
                                        increase_better=True,
                                        last=False) # 根据验证集loss选择best_net
                # true_res=[[' '.join([out_word_i2w[i] for i in valid_out_item]) for valid_out_item in valid_out]
                #                    for valid_out in valid_outs[:10]]
                # pred_res=self.predict_results(valid_ins[:10],out_i2w=out_word_i2w)
                # res_pairs=list(zip(true_res,pred_res))
                # for pair in res_pairs:
                #     print(pair)

        self._do_validation(valid_ins=valid_ins,
                            valid_outs=valid_outs,
                            increase_better=True,
                            last=True)  # 根据验证集loss选择best_net
    def predict_results(self,
                        ins,
                        out_i2w,
                        result_path=None):
        pred_out_np,_ = self.predict(ins)
        pred_texts=[]
        for pred_out_seq in pred_out_np:
            pred_equation=''
            for word in pred_out_seq:
                if word==0:
                    break
                pred_equation+=' '+out_i2w[word]
            pred_texts.append(pred_equation)
        pred_texts=[' '.join([out_i2w[i] for i in pred_out_seq[:list(pred_out_seq).index(0)]])
                        for pred_out_seq in pred_out_np]
        return pred_texts


if __name__=='__main__':
    logging.info('The main parameters:\n'
                 'in_min_token_count = {},\n'
                 'out_min_token_count = {},\n'
                 'train_data_num = {},\n'
                 'valid_data_num = {},\n'
                 'test_data_num = {},\n'
                 'model_name = {},\n'
                 'model_id = {},\n'
                 'enc_emb_dim = {},\n'
                 'dec_word_emb_dim = {},\n'
                 'enc_node_emb_path ={},\n'
                 'enc_edge_emb_path ={},\n'
                 'dec_word_emb_path = {},\n'
                 'enc_node_emb_freeze = {},\n'
                 'enc_edge_emb_freeze = {},\n'
                 'dec_word_emb_freeze = {},\n'
                 'enc_head_num = {},\n'
                 'dec_head_num = {},\n'
                 'enc_head_dim = {},\n'
                 'dec_head_dim = {},\n'
                 'enc_ff_hid_dim = {},\n'
                 'dec_ff_hid_dim = {},\n'
                 'enc_att_layer_num = {},\n'
                 'dec_att_layer_num = {},\n'
                 'enc_drop_rate = {},\n'
                 'dec_drop_rate = {},\n'
                 'batch_size = {},\n'
                 'big_epochs = {},\n'
                 'regular_rate = {},\n'
                 'lr_base = {},\n'
                 'lr_decay = {},\n'
                 'min_lr_rate = {},\n'
                 'warm_big_epochs = {}\n'.format(in_min_token_count,
                                                 out_min_token_count,
                                                 train_data_num,
                                                 valid_data_num,
                                                 test_data_num,
                                                 model_name,
                                                 model_id,
                                                 enc_emb_dim,
                                                 dec_word_emb_dim,
                                                 enc_node_emb_path,
                                                 enc_edge_emb_path,
                                                 dec_word_emb_path,
                                                 enc_node_emb_freeze,
                                                 enc_edge_emb_freeze,
                                                 dec_word_emb_freeze,
                                                 enc_head_num,
                                                 dec_head_num,
                                                 enc_head_dim,
                                                 dec_head_dim,
                                                 enc_ff_hid_dim,
                                                 dec_ff_hid_dim,
                                                 enc_att_layer_num,
                                                 dec_att_layer_num,
                                                 enc_drop_rate,
                                                 dec_drop_rate,
                                                 batch_size,
                                                 big_epochs,
                                                 regular_rate,
                                                 lr_base,
                                                 lr_decay,
                                                 min_lr_rate,
                                                 warm_big_epochs))
    logging.info('Load data ...')
    # print(train_avail_data_path)
    with codecs.open(train_avail_data_path, 'rb') as f:
        _, _, train_ins, train_outs = zip(*pickle.load(f))
    with codecs.open(valid_avail_data_path, 'rb') as f:
        _, _,valid_ins,valid_outs = zip(*pickle.load(f))
    with codecs.open(test_avail_data_path, 'rb') as f:
        _, _,test_ins,test_outs = zip(*pickle.load(f))
    # valid_ins,valid_outs,test_ins,test_outs=valid_ins[:1000],valid_outs[:1000],test_ins[:1000],test_outs[:1000]
    # print(train_outs[:10])
    with codecs.open(word_w2i_path,'rb') as f:
        out_word_w2i=pickle.load(f)
    with codecs.open(word_i2w_path,'rb') as f:
        out_word_i2w=pickle.load(f)

    # in_max_len = max([len(seqs[0]) for seqs in train_ins])  # 最大输入长度
    # out_max_len = max([len(seq) for seq in train_outs])  # 最大输出长度
    # in_node_voc_size = max([np.max(seqs[0, :]) for seqs in train_ins]) + 1  # 词表大小，词表从0开始，因此要最大序号+1才行(unknown标记也算进去了）
    # in_edge_voc_size = max([np.max(seqs[2, :]) for seqs in train_ins]) + 1  # 词表大小，词表从0开始，因此要最大序号+1才行(unknown标记也算进去了）
    # in_ast_pos_voc_size = max([np.max(seqs[[1, 3], :]) for seqs in train_ins]) + 1
    # out_word_voc_size = max(np.max(seq) for seq in train_outs) + 3  # 因为又一个begin id和一个end_id
    # out_dim = max([np.max(seq) for seq in train_outs]) + 2  # 输出维度大小，也是输出词表大小，0为<PAD>，1为<UNK>
    # test_out_max_len = max([max([len(seq) for seq in item]) for item in test_outs])

    model = TModel(model_dir=model_dir,
                   model_name=model_name,
                   model_id=model_id,
                   enc_emb_dim=enc_emb_dim,
                   dec_word_emb_dim=dec_word_emb_dim,
                   enc_node_emb_path=enc_node_emb_path,
                   enc_edge_emb_path=enc_edge_emb_path,
                   dec_word_emb_path=dec_word_emb_path,
                   enc_node_emb_freeze=enc_node_emb_freeze,
                   enc_edge_emb_freeze=enc_edge_emb_freeze,
                   dec_word_emb_freeze=dec_word_emb_freeze,
                   enc_att_layer_num=enc_att_layer_num,
                   dec_att_layer_num=dec_att_layer_num,
                   enc_drop_rate=enc_drop_rate,
                   dec_drop_rate=dec_drop_rate,
                   enc_head_num=enc_head_num,
                   enc_head_dim=enc_head_dim,
                   dec_head_num=dec_head_num,
                   dec_head_dim=dec_head_dim,
                   enc_ff_hid_dim=enc_ff_hid_dim,
                   dec_ff_hid_dim=dec_ff_hid_dim,
                   batch_size=batch_size,
                   big_epochs=big_epochs,
                   regular_rate=regular_rate,
                   lr_base=lr_base,
                   lr_decay=lr_decay,
                   min_lr_rate=min_lr_rate,
                   warm_big_epochs=warm_big_epochs,
                   Net=TransNet,
                   Dataset=Datasetx
                   )  # 初始化模型对象的一个实例
    # model.load_params()

    model.fit(train_ins=train_ins,
              train_outs=train_outs,
              out_word_w2i=out_word_w2i,  # token的映射字典
              out_word_i2w=out_word_i2w,  # toen的反映射字典
              valid_ins=valid_ins,
              valid_outs=valid_outs,
              train_metrics=train_metrics,
              valid_metric=valid_metric,
              verbose=fit_log_verbose,
              train=train
              )
    logging.info('The main parameters:\n'
                 'in_min_token_count = {},\n'
                 'out_min_token_count = {},\n'
                 'train_data_num = {},\n'
                 'valid_data_num = {},\n'
                 'test_data_num = {},\n'
                 'model_name = {},\n'
                 'model_id = {},\n'
                 'enc_emb_dim = {},\n'
                 'dec_word_emb_dim = {},\n'
                 'enc_node_emb_path ={},\n'
                 'enc_edge_emb_path ={},\n'
                 'dec_word_emb_path = {},\n'
                 'enc_node_emb_freeze = {},\n'
                 'enc_edge_emb_freeze = {},\n'
                 'dec_word_emb_freeze = {},\n'
                 'enc_head_num = {},\n'
                 'dec_head_num = {},\n'
                 'enc_head_dim = {},\n'
                 'dec_head_dim = {},\n'
                 'enc_ff_hid_dim = {},\n'
                 'dec_ff_hid_dim = {},\n'
                 'enc_att_layer_num = {},\n'
                 'dec_att_layer_num = {},\n'
                 'enc_drop_rate = {},\n'
                 'dec_drop_rate = {},\n'
                 'batch_size = {},\n'
                 'big_epochs = {},\n'
                 'regular_rate = {},\n'
                 'lr_base = {},\n'
                 'lr_decay = {},\n'
                 'min_lr_rate = {},\n'
                 'warm_big_epochs = {}\n'.format(in_min_token_count,
                                                 out_min_token_count,
                                                 train_data_num,
                                                 valid_data_num,
                                                 test_data_num,
                                                 model_name,
                                                 model_id,
                                                 enc_emb_dim,
                                                 dec_word_emb_dim,
                                                 enc_node_emb_path,
                                                 enc_edge_emb_path,
                                                 dec_word_emb_path,
                                                 enc_node_emb_freeze,
                                                 enc_edge_emb_freeze,
                                                 dec_word_emb_freeze,
                                                 enc_head_num,
                                                 dec_head_num,
                                                 enc_head_dim,
                                                 dec_head_dim,
                                                 enc_ff_hid_dim,
                                                 dec_ff_hid_dim,
                                                 enc_att_layer_num,
                                                 dec_att_layer_num,
                                                 enc_drop_rate,
                                                 dec_drop_rate,
                                                 batch_size,
                                                 big_epochs,
                                                 regular_rate,
                                                 lr_base,
                                                 lr_decay,
                                                 min_lr_rate,
                                                 warm_big_epochs))
    # #用开发集测试模型性能
    test_eval_df = model.eval_seq(test_ins=test_ins,
                                  test_outs=test_outs,
                                  test_metrics=test_metrics,
                                  )

    print('Model performance on test dataset:\n')
    print(test_eval_df.iloc[:, :4])
    print(test_eval_df.iloc[:, 4:])

    test_data = zip(test_ins, test_outs, list(range(len(test_ins))))
    import json

    # test_data = list(
    #     filter(lambda x: max([len(seq) for seq in x[1]]) < 50 and max([len(seq) for seq in x[1]]) > 35, test_data))
    test_ins, test_outs, test_ids = zip(*test_data)
    test_preds = model.predict_results(test_ins, out_i2w=out_word_i2w)
    with open(test_token_data_path, 'r') as f:
        test_token_data = json.load(f)
    # test_ins=['\n'.join(item['texts']) for item in test_token_data]
    # test_outs=['\n'.join([' '.join([out_word_i2w[i] for i in seq])
    #                      for seq in item]) for item in test_outs]
    test_outs = [' '.join([out_word_i2w[i] for i in item]) for item in test_outs]
    # content=''
    # for test_pred,test_out in zip(test_preds,test_outs):
    #     content+=test_pred+'\n'+test_out+'\n\n'
    # with open('test_2.txt','w') as f:
    #     f.write(content)
    test_result_dir=os.path.join(cur_data_dir,'test_result')
    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)
    test_result_path=os.path.join(test_result_dir,'test_result.json')
    test_result = []
    for test_pred, test_out, test_id in zip(test_preds, test_outs, test_ids):
        test_result.append({'code': test_token_data[test_id]['code'],
                             'pred_description': test_pred,
                             'ground_truth': test_token_data[test_id]['token_text']})
    with open(test_result_path, 'w') as f:
        json.dump(test_result, f, indent=4, ensure_ascii=False)
