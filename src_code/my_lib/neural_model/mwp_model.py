#coding=utf-8
from base_model import BaseModel
import sys
sys.path.append('../')
from my_evaluation import *
# sys.path.append('neural_module')
from neural_module.learn_strategy import LrWarmUp
from neural_module.transformer import TranEnc
from neural_module.embedding import PosEnc
# sys.path.append('../neural_model')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import numpy as np
import os
import logging
import pickle
from datetime import datetime
import codecs
import copy
from torchtext.data.metrics import bleu_score
import pandas as pd
import torch.nn.functional as F


class TransMWP(BaseModel):
    def save_params(self, param_path=None):
        super().save_params(param_path)
        with open(param_path,'rb') as f:
            param_dict=pickle.load(f)
        param_dict.update({'operators':self.operators,
                           'out_i2w':self.out_i2w})
        with open(param_path,'wb') as f:
            pickle.dump(param_dict,f)

    def predict(self,ins,beam_width=1):
        n_symbol = '<N>'
        o_symbol = '<O>'
        def get_sym(token):
            return o_symbol if token in self.operators else n_symbol
        logging.info('Predict outputs of %s' % self.model_name)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择GPU优先
        # self.net = self.net.to(device)  # 数据转移到设备,不重新赋值不行
        self.net.eval()
        dataset = self.Dataset(ins,
                               in_max_len=self.in_max_len,
                               out_max_len=self.out_max_len,
                               out_begin_idx=self.out_begin_idx)
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                            num_workers=8)
        pred_out_probs = []
        pred_out_labels = []
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') #选择GPU优先
        # beam_width = 1
        with torch.no_grad():
            # pred_out_probs = []
            for batch_features, batch_out_inputs in data_loader:
                batch_features = batch_features.to(device)  #(B,L_x)
                batch_out_inputs = batch_out_inputs.to(device)  #(B,L_y)
                # print(batch_features.size())
                # batch_encoder=self.net.module.encoding(batch_features) #(B,L_x,D_x)
                batch_encoder=self.net.module.encoder(batch_features) #(B,L_x,D_x)
                for i in range(batch_features.size(0)):
                    # beam_width = 10
                    com_i_out_probs=[]
                    com_mean_i_log_probs=[]
                    tmp_i_encoder=batch_encoder[i,:,:].unsqueeze(0) #(1,L_x,D_x)
                    tmp_i_out_inputs=batch_out_inputs[i,:].unsqueeze(0) #(1,L_y)
                    tmp_i_out_probs = torch.zeros((1,self.out_dim,self.out_max_len,)).to(device)   #(BW,D_y,L_y)
                    tmp_acc_i_log_probs=torch.zeros((1,)).to(device)   #(BW,)
                    # tmp_mean_i_log_probs=torch.zeros((1,)).to(device)   #(BW,)
                    # tmp_lens=np.zeros((beam_width,))    #(BW,)
                    aux_stacks = [[]]
                    for j in range(self.out_max_len):
                        # print(tmp_i_out_inputs.size(),tmp_i_encoder.size(),tmp_i_encoder.expand(tmp_i_out_inputs.size(0),-1,-1).size())
                        pred_i_out_probs = self.net.module.decoder(tmp_i_out_inputs, tmp_i_encoder.expand(tmp_i_out_inputs.size(0),-1,-1))  # (1/BW,D_y,L_y)
                        # pred_i_out_probs = self.net.module.decoding(tmp_i_out_inputs, tmp_i_encoder.expand(tmp_i_out_inputs.size(0),-1,-1))  # (1/BW,D_y,L_y)
                        tmp_i_out_probs[:,:,j]=pred_i_out_probs[:,:,j]  #(1/BW,D_y,L_y)
                        tmp_i_out_probs=tmp_i_out_probs.unsqueeze(1).expand(-1,beam_width,-1,-1)    #(1/BW,BW,D_y,L_y)
                        tmp_i_out_probs=tmp_i_out_probs.contiguous().view(-1,self.out_dim,self.out_max_len)   #(1/BW*BW,D_y,L_y)

                        pred_ij_out_log_probs=F.softmax(pred_i_out_probs[:,:,j],1)    #(1/BW,D_y,L_y)
                        topk_ij_out_log_probs,_=pred_ij_out_log_probs.topk(beam_width,dim=1)    #(1/BW,BW)
                        topk_ij_out_log_probs=topk_ij_out_log_probs.contiguous().view(-1)    #(1/BW*BW,)

                        tmp_acc_i_log_probs=tmp_acc_i_log_probs.unsqueeze(1).expand(-1,beam_width).contiguous().view(-1)   #(1/BW*BW,)
                        # print(tmp_acc_i_log_probs.size(),topk_ij_out_log_probs.size())
                        tmp_acc_i_log_probs=tmp_acc_i_log_probs.add(topk_ij_out_log_probs)  #(1/BW*BW,)
                        tmp_acc_i_log_probs,topk_indices=tmp_acc_i_log_probs.topk(beam_width)   #(BW,)
                        tmp_mean_i_log_probs=tmp_acc_i_log_probs/(j+1) #(BW,)

                        # topk_ij_out_log_probs=topk_ij_out_log_probs[topk_indices,:,:]   #(BW,
                        # print(tmp_i_out_probs.size())
                        tmp_i_out_probs=tmp_i_out_probs[topk_indices,:,:]   #(BW,D_y,L_y)

                        tmp_i_out_inputs=tmp_i_out_inputs.unsqueeze(1).expand(-1,beam_width,-1).contiguous().view(-1,self.out_max_len)    #(1/BW*BW,L_y)
                        tmp_i_out_inputs=tmp_i_out_inputs[topk_indices,:]   #(BW,L_y)
                        # print(tmp_i_out_inputs.size())
                        pred_ij_out_indices=torch.argmax(tmp_i_out_probs[:,1:,j],dim=1)+1 #(BW,)
                        if j<self.out_max_len - 1:
                            tmp_i_out_inputs[:,j+1]=pred_ij_out_indices
                        pred_ij_out_indices=pred_ij_out_indices.cpu().data.numpy()
                        # print(tmp_i_out_inputs.size())

                        # topk_indices=topk_indices
                        # print(pred_ij_out_indices)
                        # aux_stacks=np.expand_dims(np.array(aux_stacks),-1).repeat(beam_width,axis=-1).reshape(-1)  #(BW*BW,)
                        # aux_stacks=aux_stacks[topk_indices.cpu().data.numpy()].tolist()    #(BW,)
                        # if j==0:
                        #     aux_stacks=[[]]*beam_width
                        # else:
                        tmp_aux_stacks=[]
                        for aux_stack in aux_stacks:
                            tmp_aux_stacks+=[aux_stack]*beam_width
                        aux_stacks=[tmp_aux_stacks[topk_idx] for topk_idx in topk_indices.cpu().data.numpy()]

                        for m,(aux_stack,pred_ij_out_idx) in enumerate(zip(aux_stacks,pred_ij_out_indices.tolist())):
                            symbol = get_sym(self.out_i2w[pred_ij_out_idx])
                            aux_stacks[m].append(symbol)
                            break_flag=False
                            if j == 0 and symbol == n_symbol:  # 如果第一个就是数字
                                break_flag=True
                            elif len(aux_stacks[m]) > 2:
                                while len(aux_stacks[m]) > 2 and aux_stacks[m][-1] == n_symbol and aux_stacks[m][-2] == n_symbol:
                                    aux_stacks[m].pop(-1)
                                    aux_stacks[m].pop(-1)
                                    aux_stacks[m][-1] = n_symbol
                                if len(aux_stacks[m]) == 1:
                                    break_flag = True

                            if break_flag:
                                com_i_out_probs.append(tmp_i_out_probs[m, :, :].to('cpu').data.numpy())  # (D_y,L_y)
                                com_mean_i_log_probs.append(tmp_mean_i_log_probs[m].to('cpu').data.numpy())  # (1,)
                                tmp_i_out_inputs=tmp_i_out_inputs[torch.arange(tmp_i_out_inputs.size(0))!=m]    #去掉m这个
                                tmp_i_out_probs=tmp_i_out_probs[torch.arange(tmp_i_out_probs.size(0))!=m]    #去掉m这个
                                tmp_acc_i_log_probs=tmp_acc_i_log_probs[torch.arange(tmp_acc_i_log_probs.size(0))!=m]    #去掉m这个
                                # tmp_mean_i_log_probs=tmp_mean_i_log_probs[torch.arange(tmp_mean_i_log_probs.size(0))!=m]    #去掉m这个
                                # beam_width-=1
                                break
                        if tmp_i_out_probs.size(0)==0:
                            break
                    if len(com_i_out_probs)>0:
                        max_mean_i_log_prob_idx=com_mean_i_log_probs.index(max(com_mean_i_log_probs))   #(value)
                        # print(com_mean_i_log_probs)
                        max_com_i_out_probs=com_i_out_probs[max_mean_i_log_prob_idx][np.newaxis, :, :]#[(1,D_y,L_y)]
                        i_out_length=int(np.sign(np.abs(max_com_i_out_probs).sum(1)).sum(1)[0])
                        max_com_i_out_labels=np.argmax(max_com_i_out_probs[:,1:,:],axis=1)+1
                        max_com_i_out_labels[:,i_out_length:]=0
                        pred_out_probs.append(max_com_i_out_probs)   #[(1,D_y,L_y)]
                        pred_out_labels.append(max_com_i_out_labels)    #[(1,D_y)]
                    else:
                        max_mean_i_log_prob_idx=torch.argmax(tmp_mean_i_log_probs)  #(tensor value)
                        max_com_i_out_probs = tmp_i_out_probs[max_mean_i_log_prob_idx,:,:].unsqueeze(0).to('cpu').data.numpy()  # [(1,D_y,L_y)]
                        i_out_length = int(np.sign(np.abs(max_com_i_out_probs).sum(1)).sum(1)[0])
                        max_com_i_out_labels = np.argmax(max_com_i_out_probs[:,1:,:],axis=1)+1
                        max_com_i_out_labels[:, i_out_length:] = 0
                        pred_out_probs.append(max_com_i_out_probs)  # [(1,D_y,L_y)]
                        pred_out_labels.append(max_com_i_out_labels)  # [(1,D_y)]

        pred_out_prob_np = np.concatenate(pred_out_probs, axis=0)  # (B+,D,L_y)
        pred_out_label_np = np.concatenate(pred_out_labels, axis=0)  # (B+,L_y)
        self.net.train()
        return pred_out_label_np,pred_out_prob_np

    # def predict(self,ins):
    #     '''
    #     预测样本的类别标签的接口
    #     :param ins: 样本特征集二维数组
    #     :return: 预测出的类别标签一维数组,或值
    #     '''
    #     n_symbol = '<N>'
    #     o_symbol = '<O>'
    #     def get_sym(token):
    #         return o_symbol if token in self.operators else n_symbol
    #     pred_out_np,pred_out_prob_np=super().predict(ins)
    #     # print(pred_out_np[:3,:])
    #     for i,pred_outs in enumerate(pred_out_np):
    #         aux_stack=[]
    #         for j,out_idx in enumerate(pred_outs):    #prefix
    #             # print('inner_j:', j)
    #             token=self.out_i2w[out_idx]
    #             symbol=get_sym(token)
    #             aux_stack.append(symbol)
    #             if j==0 and symbol==n_symbol:    #如果第一个就是数字
    #                 break
    #             elif len(aux_stack)>2:
    #                 while len(aux_stack)>2 and aux_stack[-1]==n_symbol and aux_stack[-2]==n_symbol:
    #                     aux_stack.pop(-1)
    #                     aux_stack.pop(-1)
    #                     aux_stack[-1]=n_symbol
    #                 if len(aux_stack)==1:
    #                     break
    #             # print('inner_j:',j)
    #         # for j,out_idx in enumerate(pred_outs):  #postfix
    #         #     token=self.out_i2w[out_idx]
    #         #     symbol=get_sym(token)
    #         #     aux_stack.append(symbol)
    #         #     # if j==0 and symbol==n_symbol:    #如果第一个就是数字
    #         #     #     break
    #         #     if len(aux_stack)>2:
    #         #         while len(aux_stack)>2 and aux_stack[-1]==o_symbol and aux_stack[-2]==n_symbol and aux_stack[-3]==n_symbol:
    #         #             aux_stack.pop(-1)
    #         #             aux_stack.pop(-1)
    #         #             # aux_stack[-1]=n_symbol
    #         #         if len(aux_stack)==1:
    #         #             break
    #         # print('outer_j:',j)
    #         pred_out_np[i,j+1:]=0
    #         # if len(aux_stack)>1 and j==len(pred_outs)-1:
    #         #     pred_out_np[i,1:]=0
    #     return pred_out_np,pred_out_prob_np\


class RNNMWP(TransMWP):
    # def predict(self,ins):
    #     n_symbol = '<N>'
    #     o_symbol = '<O>'
    #     def get_sym(token):
    #         return o_symbol if token in self.operators else n_symbol
    #     logging.info('Predict outputs of %s' % self.model_name)
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择GPU优先
    #     # self.net = self.net.to(device)  # 数据转移到设备,不重新赋值不行
    #     self.net.eval()
    #     dataset = self.Dataset(ins,
    #                            in_max_len=self.in_max_len,
    #                            out_max_len=self.out_max_len,
    #                            out_begin_idx=self.out_begin_idx)
    #     data_loader = DataLoader(dataset=dataset,
    #                              batch_size=self.batch_size,
    #                              shuffle=False)
    #     pred_out_probs = []
    #     pred_out_labels = []
    #     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') #选择GPU优先
    #     beam_width = 1
    #     with torch.no_grad():
    #         # pred_out_probs = []
    #         for batch_features, batch_out_inputs in data_loader:
    #             batch_features = batch_features.to(device)  #(B,L_x)
    #             batch_out_inputs = batch_out_inputs.to(device)  #(B,L_y)
    #             # print(batch_features.size())
    #             # batch_encoder=self.net.module.encoding(batch_features) #(B,L_x,D_x)
    #             batch_encoder=self.net.module.encoder(batch_features) #(B,L_x,D_x)
    #             for i in range(batch_features.size(0)):
    #                 # beam_width = 10
    #                 com_i_out_probs=[]
    #                 com_mean_i_log_probs=[]
    #                 tmp_i_encoder=batch_encoder[i,:,:].unsqueeze(0) #(1,L_x,D_x)
    #                 tmp_i_out_inputs=batch_out_inputs[i,:].unsqueeze(0) #(1,L_y)
    #                 tmp_i_out_probs = torch.zeros((1,self.out_dim,self.out_max_len,)).to(device)   #(BW,D_y,L_y)
    #                 tmp_acc_i_log_probs=torch.zeros((1,)).to(device)   #(BW,)
    #                 # tmp_mean_i_log_probs=torch.zeros((1,)).to(device)   #(BW,)
    #                 # tmp_lens=np.zeros((beam_width,))    #(BW,)
    #                 aux_stacks = [[]]
    #                 for j in range(self.out_max_len):
    #                     # print(tmp_i_out_inputs.size(),tmp_i_encoder.size(),tmp_i_encoder.expand(tmp_i_out_inputs.size(0),-1,-1).size())
    #                     pred_i_out_probs = self.net.module.decoder(tmp_i_out_inputs, tmp_i_encoder.expand(tmp_i_out_inputs.size(0),-1,-1))  # (1/BW,D_y,L_y)
    #                     # pred_i_out_probs = self.net.module.decoding(tmp_i_out_inputs, tmp_i_encoder.expand(tmp_i_out_inputs.size(0),-1,-1))  # (1/BW,D_y,L_y)
    #                     tmp_i_out_probs[:,:,j]=pred_i_out_probs[:,:,j]  #(1/BW,D_y,L_y)
    #                     tmp_i_out_probs=tmp_i_out_probs.unsqueeze(1).expand(-1,beam_width,-1,-1)    #(1/BW,BW,D_y,L_y)
    #                     tmp_i_out_probs=tmp_i_out_probs.contiguous().view(-1,self.out_dim,self.out_max_len)   #(1/BW*BW,D_y,L_y)
    #
    #                     pred_ij_out_log_probs=F.softmax(pred_i_out_probs[:,:,j],1)    #(1/BW,D_y,L_y)
    #                     topk_ij_out_log_probs,_=pred_ij_out_log_probs.topk(beam_width,dim=1)    #(1/BW,BW)
    #                     topk_ij_out_log_probs=topk_ij_out_log_probs.contiguous().view(-1)    #(1/BW*BW,)
    #
    #                     tmp_acc_i_log_probs=tmp_acc_i_log_probs.unsqueeze(1).expand(-1,beam_width).contiguous().view(-1)   #(1/BW*BW,)
    #                     # print(tmp_acc_i_log_probs.size(),topk_ij_out_log_probs.size())
    #                     tmp_acc_i_log_probs=tmp_acc_i_log_probs.add(topk_ij_out_log_probs)  #(1/BW*BW,)
    #                     tmp_acc_i_log_probs,topk_indices=tmp_acc_i_log_probs.topk(beam_width)   #(BW,)
    #                     tmp_mean_i_log_probs=tmp_acc_i_log_probs/(j+1) #(BW,)
    #
    #                     # topk_ij_out_log_probs=topk_ij_out_log_probs[topk_indices,:,:]   #(BW,
    #                     # print(tmp_i_out_probs.size())
    #                     tmp_i_out_probs=tmp_i_out_probs[topk_indices,:,:]   #(BW,D_y,L_y)
    #
    #                     tmp_i_out_inputs=tmp_i_out_inputs.unsqueeze(1).expand(-1,beam_width,-1).contiguous().view(-1,self.out_max_len)    #(1/BW*BW,L_y)
    #                     tmp_i_out_inputs=tmp_i_out_inputs[topk_indices,:]   #(BW,L_y)
    #                     # print(tmp_i_out_inputs.size())
    #                     pred_ij_out_indices=torch.argmax(tmp_i_out_probs[:,1:,j],dim=1)+1 #(BW,)
    #                     if j<self.out_max_len - 1:
    #                         tmp_i_out_inputs[:,j+1]=pred_ij_out_indices
    #                     pred_ij_out_indices=pred_ij_out_indices.cpu().data.numpy()
    #                     # print(tmp_i_out_inputs.size())
    #
    #                     # topk_indices=topk_indices
    #                     # print(pred_ij_out_indices)
    #                     # aux_stacks=np.expand_dims(np.array(aux_stacks),-1).repeat(beam_width,axis=-1).reshape(-1)  #(BW*BW,)
    #                     # aux_stacks=aux_stacks[topk_indices.cpu().data.numpy()].tolist()    #(BW,)
    #                     # if j==0:
    #                     #     aux_stacks=[[]]*beam_width
    #                     # else:
    #                     tmp_aux_stacks=[]
    #                     for aux_stack in aux_stacks:
    #                         tmp_aux_stacks+=[aux_stack]*beam_width
    #                     aux_stacks=[tmp_aux_stacks[topk_idx] for topk_idx in topk_indices.cpu().data.numpy()]
    #
    #                     for m,(aux_stack,pred_ij_out_idx) in enumerate(zip(aux_stacks,pred_ij_out_indices.tolist())):
    #                         symbol = get_sym(self.out_i2w[pred_ij_out_idx])
    #                         aux_stacks[m].append(symbol)
    #                         break_flag=False
    #                         if j == 0 and symbol == n_symbol:  # 如果第一个就是数字
    #                             break_flag=True
    #                         elif len(aux_stacks[m]) > 2:
    #                             while len(aux_stacks[m]) > 2 and aux_stacks[m][-1] == n_symbol and aux_stacks[m][-2] == n_symbol:
    #                                 aux_stacks[m].pop(-1)
    #                                 aux_stacks[m].pop(-1)
    #                                 aux_stacks[m][-1] = n_symbol
    #                             if len(aux_stacks[m]) == 1:
    #                                 break_flag = True
    #
    #                         if break_flag:
    #                             com_i_out_probs.append(tmp_i_out_probs[m, :, :].to('cpu').data.numpy())  # (D_y,L_y)
    #                             com_mean_i_log_probs.append(tmp_mean_i_log_probs[m].to('cpu').data.numpy())  # (1,)
    #                             tmp_i_out_inputs=tmp_i_out_inputs[torch.arange(tmp_i_out_inputs.size(0))!=m]    #去掉m这个
    #                             tmp_i_out_probs=tmp_i_out_probs[torch.arange(tmp_i_out_probs.size(0))!=m]    #去掉m这个
    #                             tmp_acc_i_log_probs=tmp_acc_i_log_probs[torch.arange(tmp_acc_i_log_probs.size(0))!=m]    #去掉m这个
    #                             # tmp_mean_i_log_probs=tmp_mean_i_log_probs[torch.arange(tmp_mean_i_log_probs.size(0))!=m]    #去掉m这个
    #                             # beam_width-=1
    #                             break
    #                     if tmp_i_out_probs.size(0)==0:
    #                         break
    #                 if len(com_i_out_probs)>0:
    #                     max_mean_i_log_prob_idx=com_mean_i_log_probs.index(max(com_mean_i_log_probs))   #(value)
    #                     # print(com_mean_i_log_probs)
    #                     max_com_i_out_probs=com_i_out_probs[max_mean_i_log_prob_idx][np.newaxis, :, :]#[(1,D_y,L_y)]
    #                     i_out_length=int(np.sign(np.abs(max_com_i_out_probs).sum(1)).sum(1)[0])
    #                     max_com_i_out_labels=np.argmax(max_com_i_out_probs[:,1:,:],axis=1)+1
    #                     max_com_i_out_labels[:,i_out_length:]=0
    #                     pred_out_probs.append(max_com_i_out_probs)   #[(1,D_y,L_y)]
    #                     pred_out_labels.append(max_com_i_out_labels)    #[(1,D_y)]
    #                 else:
    #                     max_mean_i_log_prob_idx=torch.argmax(tmp_mean_i_log_probs)  #(tensor value)
    #                     max_com_i_out_probs = tmp_i_out_probs[max_mean_i_log_prob_idx,:,:].unsqueeze(0).to('cpu').data.numpy()  # [(1,D_y,L_y)]
    #                     i_out_length = int(np.sign(np.abs(max_com_i_out_probs).sum(1)).sum(1)[0])
    #                     max_com_i_out_labels = np.argmax(max_com_i_out_probs[:,1:,:],axis=1)+1
    #                     max_com_i_out_labels[:, i_out_length:] = 0
    #                     pred_out_probs.append(max_com_i_out_probs)  # [(1,D_y,L_y)]
    #                     pred_out_labels.append(max_com_i_out_labels)  # [(1,D_y)]
    #
    #     pred_out_prob_np = np.concatenate(pred_out_probs, axis=0)  # (B+,D,L_y)
    #     pred_out_label_np = np.concatenate(pred_out_labels, axis=0)  # (B+,L_y)
    #     self.net.train()
    #     return pred_out_label_np,pred_out_prob_np

    def predict(self,ins):
        '''
        预测样本的类别标签的接口
        :param ins: 样本特征集二维数组
        :return: 预测出的类别标签一维数组,或值
        '''
        n_symbol = '<N>'
        o_symbol = '<O>'
        def get_sym(token):
            return o_symbol if token in self.operators else n_symbol
        # pred_out_np,pred_out_prob_np=super().predict(ins)
        logging.info('Predict outputs of %s' % self.model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择GPU优先
        self.net = self.net.to(device)  # 数据转移到设备,不重新赋值不行
        self.net.eval()
        dataset = self.Dataset(ins,
                               in_max_len=self.in_max_len,
                               out_max_len=self.out_max_len,
                               out_begin_idx=self.out_begin_idx)
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                            num_workers=8)
        pred_out_prob_batches = []
        # batch_pred_outs = []
        with torch.no_grad():
            for batch_features, batch_out_inputs in data_loader:
                batch_features = batch_features.to(device)
                batch_out_inputs = batch_out_inputs.to(device)
                tmp_batch_outs = self.net(batch_features,batch_out_inputs,tf_rate=0) #[B,D,L)
                pred_out_prob_batches.append(tmp_batch_outs.to('cpu').data.numpy())  # [(B,D,L)]
                # batch_pred_outs.append(batch_out_inputs.to('cpu').data.numpy())
        self.net.train()

        pred_out_prob_np = np.concatenate(pred_out_prob_batches, axis=0)  # (B+,D,L)
        pred_out_np = np.argmax(pred_out_prob_np[:, 1:, :], axis=1) + 1  # (B+,L)
        # print(pred_out_np[:3,:])
        for i,pred_outs in enumerate(pred_out_np):
            aux_stack=[]
            for j,out_idx in enumerate(pred_outs):    #prefix
                # print('inner_j:', j)
                token=self.out_i2w[out_idx]
                symbol=get_sym(token)
                aux_stack.append(symbol)
                if j==0 and symbol==n_symbol:    #如果第一个就是数字
                    break
                elif len(aux_stack)>2:
                    while len(aux_stack)>2 and aux_stack[-1]==n_symbol and aux_stack[-2]==n_symbol:
                        aux_stack.pop(-1)
                        aux_stack.pop(-1)
                        aux_stack[-1]=n_symbol
                    if len(aux_stack)==1:
                        break
                # print('inner_j:',j)
            # for j,out_idx in enumerate(pred_outs):  #postfix
            #     token=self.out_i2w[out_idx]
            #     symbol=get_sym(token)
            #     aux_stack.append(symbol)
            #     # if j==0 and symbol==n_symbol:    #如果第一个就是数字
            #     #     break
            #     if len(aux_stack)>2:
            #         while len(aux_stack)>2 and aux_stack[-1]==o_symbol and aux_stack[-2]==n_symbol and aux_stack[-3]==n_symbol:
            #             aux_stack.pop(-1)
            #             aux_stack.pop(-1)
            #             # aux_stack[-1]=n_symbol
            #         if len(aux_stack)==1:
            #             break
            # print('outer_j:',j)
            pred_out_np[i,j+1:]=0
            # if len(aux_stack)>1 and j==len(pred_outs)-1:
            #     pred_out_np[i,1:]=0
        return pred_out_np,pred_out_prob_np

if __name__=='__main__':
    print(BaseModel.__dict__)
    if 'name' in BaseModel.__dict__:
        print('True')

    import codecs
    import pickle
    data_dir = '../../data/SEMEVAL2017T1/'
    avail_dataset_dir = os.path.join(data_dir, 'avail_dataset/')
    train_data_name = 'train_data.pkl'
    test_data_name = 'test_data.pkl'
    dev_data_name = 'dev_data.pkl'
    model_dir = os.path.join(data_dir, 'model/')
    # tensorboard_dir='../../../data/sample_data/model/tensorboard_log/'
    logging.info('Load data ...')
    with codecs.open(os.path.join(avail_dataset_dir, train_data_name), 'rb') as f:
        train_outs, train_features_1, train_features_2 = zip(*pickle.load(f))
        train_features = list(zip(train_features_1, train_features_2))
        # print(train_outs)

    train_dataset=PairDataset(train_features,train_outs)
    train_loader=DataLoader(dataset=train_dataset,
                            batch_size=32,
                            shuffle=True,
                            num_workers=8)
    for epoch in range(2):
        for i,(train_feature_tensor1,train_feature_tensor2,train_out_tensor) in enumerate(train_loader):
            print('epoch:',epoch)
            print('>>>>>train_feature_tensor1:',train_feature_tensor1)
            print('>>>>>train_feature_tensor2:', train_feature_tensor2)
            print('>>>>>train_out_tensor:', train_out_tensor)