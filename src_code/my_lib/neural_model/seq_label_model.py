#coding=utf-8
from .base_model import BaseModel,BaseNet
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

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Datasetx(Dataset):
    '''
    文本对数据集对象（根据具体数据再修改）
    '''
    def __init__(self,ins,outs=None,in_max_len=None):
        self.len=len(ins)
        self.in_max_len=in_max_len
        if in_max_len is None:
            self.in_max_len = max([len(seq) for seq in ins])
        self.ins=ins
        self.outs=outs
    def __getitem__(self, index):
        tru_feature=self.ins[index][:self.in_max_len] #先做截断
        pad_feature = np.lib.pad(tru_feature, (0, self.in_max_len - len(tru_feature)),
                                        'constant', constant_values=(0, 0))  # padding
        if self.outs is None:
            return torch.tensor(pad_feature)
        else:
            tru_out=self.outs[index][:self.in_max_len] #先做截断
            pad_out=np.lib.pad(tru_out, (0, self.in_max_len - len(tru_out)),
                                        'constant', constant_values=(0, 0))  # padding
            return torch.tensor(pad_feature),\
                   torch.tensor(pad_out).long()

    def __len__(self):
        return self.len

class TransNet(BaseNet):
    def __init__(self,
                 in_max_len,
                 vocab_size,
                 out_dim,
                 embed_dim=300,
                 token_init_embed=None,
                 token_embed_freeze=False,
                 att_layer_num=6,
                 head_num=10,
                 head_dim=None,
                 drop_rate=0.
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        #获取Net的init参数
        self.init_params=locals()
        del self.init_params['self']
        self.position_encoding = PosEnc(max_len=in_max_len,embed_dim=embed_dim,train=True)
        if token_init_embed is None:
            self.token_embedding=nn.Embedding(vocab_size,embed_dim,padding_idx=0)
            nn.init.xavier_uniform_(self.token_embedding.weight[1:,:])  #nn.init.xavier_uniform_
            # self.position_encoder.weight.data[0, :] = 0  # 上面初始化后padding0被黑了，靠
        else:
            # print(token_init_embed.size())
            # assert token_init_embed.shape==(vocab_size,embed_dim)
            token_init_embed=torch.tensor(token_init_embed,dtype=torch.float32)
            self.token_embedding=nn.Embedding.from_pretrained(token_init_embed,freeze=token_embed_freeze,padding_idx=0)

        self.encoder = TranEnc(query_dim=embed_dim,
                               head_num=head_num,
                               head_dim=head_dim,
                               layer_num=att_layer_num,
                               drop_rate=drop_rate)

        # self.linear = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        # self.linear1 = nn.Linear(embed_dim,128)
        # self.linear2 = nn.Linear(128, out_dim)
        # self.relu = nn.ReLU()
        self.out_fc = nn.Sequential(
            # nn.Dropout(drop_rate),
            nn.Linear(embed_dim, 128),  # 4
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(128, out_dim),
        )
        self.dropout = nn.Dropout(p=drop_rate)
        # if self.out_dim>1:
        #     self.softmax=nn.Softmax(dim=-1)

    # def get_init_params(self):
    #     if 'self' in self.init_params:
    #         del self.init_params['self']
    #     return self.init_params

    def forward(self, x):
        '''

        :param x: [B,L]
        :return:
        '''
        token_embed=self.token_embedding(x)*np.sqrt(self.embed_dim)    #(B,L,D)
        pos_embed=self.position_encoding(x) #(B,L,D)

        token_mask = x.abs().sign()  # (B,L)
        token_coder = self.layer_norm(token_embed.add(pos_embed))  # (B,L,D)
        # token_coder = token_coder.mul(token_mask.unsqueeze(-1).expand(-1, -1, token_coder.size(-1)).float())
        # token_coder = self.linear(token_coder)  # (B,L,D)

        token_coder=self.dropout(token_coder) #(B,L,D)

        token_coder=self.encoder(token_coder,token_mask)   #[2,],(B,L,D),(B,D)
        # token_coder = token_coder.mul(token_mask.unsqueeze(-1).expand(-1, -1, token_coder.size(-1)).float())

        outputs=self.out_fc(token_coder)     #(B,L,out_dim)

        return outputs.transpose(1, 2)  #(B,out_dim,L)


class TransSeqLabel(BaseModel):
    def __init__(self,
                 model_dir,
                 model_name='Transformer_based_model',
                 model_id=None,
                 embed_dim=512,
                 token_embed_path=None,
                 token_embed_freeze=True,
                 head_num=8,
                 head_dim=None,
                 att_layer_num=6,
                 drop_rate=0.3,
                 batch_size=32,
                 big_epochs=20,
                 regular_rate=1e-5,
                 lr_base=0.001,
                 lr_decay=0.9,
                 min_lr_rate=0.01,
                 warm_big_epochs=2,
                 Net=TransNet,
                 Dataset=Datasetx,
                 ):
        '''
        构造函数
        :param model_dir: 模型存放目录
        :param model_name: 模型名称
        :param model_id: 模型id
        :param max_class_num: 一个apk里最大类数量
        :param embed_dim: 词向量维度
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
        self.init_params = locals()
        super().__init__(model_name=model_name,
                         model_dir=model_dir,
                         model_id=model_id)

        # self.Dataset=Datasetx
        self.embed_dim = embed_dim
        self.token_embed_path = token_embed_path
        self.token_embed_freeze = token_embed_freeze
        self.head_num = head_num
        self.head_dim=head_dim
        self.att_layer_num = att_layer_num
        self.drop_rate = drop_rate
        self.batch_size = batch_size
        self.big_epochs = big_epochs
        self.regular_rate=regular_rate
        self.lr_base = lr_base
        self.lr_decay = lr_decay
        self.min_lr_rate=min_lr_rate
        self.warm_big_epochs=warm_big_epochs
        self.Net = Net
        self.Dataset = Dataset

    def fit(self,
            train_features,
            train_outs,
            out2tag=None,
            tag2span_func=None,
            valid_features=None,
            valid_outs=None,
            train_metrics=[get_overall_accuracy],
            valid_metric=get_overall_accuracy,
            verbose=0
            ):
        # print(self.__dict__)
        '''
        训练模型接口
        :param train_features: 特征集，结构为[[文本ndarray,文本ndarray],[文本ndarray,文本ndarray],...],双层list+ndarray文本对
        :param train_outs: 输出标记
        :param use_tensorboard: 是否使用tensorboard
        :param verbose: 训练时显示日志信息，为0表示不输出日志，为1每个batch都输出，为2表示每遍历完一次所有数据输出一次
        :param train_metrics: 一个列表的用于训练时对训练数据的评价函数
        :param valid_metrics: 用于训练时对验证数据的评价函数
        :return:
        '''
        logging.info('Train %s' % self.model_name)
        self.out2tag=out2tag
        self.tag2span_func=tag2span_func
        self.train_metrics = train_metrics
        self.valid_metric = valid_metric
        # torch.autograd.set_detect_anomaly(True)
        self.in_max_len = max(len(seq) for seq in train_features)  # 最大序列长度
        # print(self.in_max_len)
        self.vocab_size = max(np.max(seq) for seq in train_features) + 1  # 词表大小，词表从0开始，因此要最大序号+1才行(unknown标记也算进去了）
        # print(self.vocab_size)
        # self.out_dim = len(np.unique(np.array(train_outs)))  # 输出节点个数,即不同类别的out
        # print(self.out_dim)
        self.sort_unique_outs = sorted(list(np.unique(np.concatenate(train_outs))))  # 排序的unique outs
        # print(self.sort_unique_outs)
        self.out_dim=len(self.sort_unique_outs)+1  #因为有0，所以要+1

        # print(self.sort_unique_outs)
        token_embed_weight = None
        if self.token_embed_path is not None:  # 如果加载预训练词向量
            token_embed_weight = np.load(self.token_embed_path)
            # print(token_embed_weight[-1,:])
            # print(np.linalg.norm(token_embed_weight, axis=1, keepdims=True)[0,:])
            # token_embed_weight[1:,:]/=np.linalg.norm(token_embed_weight[1:,:], axis=1, keepdims=True)    #归一化,第1行都是0不能参加运算
            # print(token_embed_weight[-1, :])
            self.vocab_size = token_embed_weight.shape[0]
            # print(token_embed_weight[2,:])
            # print(token_embed_weight.shape)
        # print()
        net = self.Net(in_max_len=self.in_max_len,
                       vocab_size=self.vocab_size,
                       embed_dim=self.embed_dim,
                       token_init_embed=token_embed_weight,
                       token_embed_freeze=self.token_embed_freeze,
                       att_layer_num=self.att_layer_num,
                       head_num=self.head_num,
                       head_dim=self.head_dim,
                       out_dim=self.out_dim,
                       drop_rate=self.drop_rate,
                       )

        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') #选择GPU优先
        self.net = nn.DataParallel(net.to(device))  # 并行使用多GPU
        # self.net.to(device) #数据转移到设备

        self.net.train()

        # for p in self.net.parameters(): #初始化非embedding层的参数
        #     if p.size(0) != self.vocab_size:
        #         nn.init.normal_(p, 0.0, 0.05)

        # self.best_net=None
        self.optimizer = optim.Adam(self.net.parameters(),
                               lr=self.lr_base,
                               weight_decay=self.regular_rate)
        if self.token_embed_path is not None:  # 如果加载预训练词向量
            token_embed_param = [x for x in self.net.parameters() if x.requires_grad and x.size(0) == self.vocab_size]
            # print(len(token_embed_param))
            ex_param = [x for x in self.net.parameters() if x.requires_grad and x.size(0) != self.vocab_size]
            optim_cfg = [{'params': token_embed_param, 'lr': self.lr_base*0.1},
                         {'params': ex_param, 'lr': self.lr_base, 'weight_decay': self.regular_rate}, ]
            self.optimizer = optim.Adam(optim_cfg)
        # optimizer=optim.SGD(self.net.parameters(),lr=self.lr_base,weight_decay=self.regular_rate,momentum=0.9)

        # scheduler=lr_scheduler.ExponentialLR(optimizer,gamma=self.lr_decay)

        # scheduler=lr_scheduler.ReduceLROnPlateau(optimizer)

        # min_lr=self.lr_base*0.1
        # scheduler=lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                          mode='min',
        #                                          factor=self.lr_decay,
        #                                          patience=1,
        #                                          verbose=False,
        #                                          threshold=1e-4,
        #                                          threshold_mode='rel',
        #                                          cooldown=0,
        #                                          min_lr=min_lr,
        #                                          eps=10e-8)

        # criterion=nn.MSELoss(reduction='mean')
        # if self.out_dim>1:
        criterion=nn.CrossEntropyLoss(reduction='mean',ignore_index=0)
        self.criterion = nn.DataParallel(CriterionNet(criterion).to(device))

        # train_features=sorted(train_features,key=lambda x: len(x))  #根据文本长度对数据集排序
        train_set = self.Dataset(train_features, train_outs)
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                            num_workers=8)

        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
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
        if self.out2tag is None:
            self.seq_mode='POS'
        else:
            self.seq_mode='NER'
        for i in range(self.big_epochs):
            for j, (batch_features, batch_outs) in enumerate(train_loader):
                batch_features=batch_features.to(device)
                batch_outs=batch_outs.to(device)

                pred_outs=self.net(batch_features)
                loss=self.criterion(pred_outs,batch_outs)
                self.optimizer.zero_grad()
                loss.backward()
                # clip_grad_norm(self.net.parameters(),1e-4)  #减弱梯度爆炸
                self.optimizer.step()

                self.scheduler.step()
                # logging.info('The learning rates of first and last parameter group:{}, {}'.
                #              format(optimizer.param_groups[0]['lr'], optimizer.param_groups[-1]['lr']))                # print(scheduler.get_lr()[0])
                self._log_fit_eval(loss=loss,
                                   big_step=i+1,
                                   batch_step=j+1,
                                   big_epochs=self.big_epochs,
                                   batch_epochs=len(train_loader),
                                   pred_outs=pred_outs,
                                   true_outs=batch_outs,
                                   seq_mode=self.seq_mode,
                                   verbose=verbose)
                # print(train_loader.__len__()//2,j)
                # if j==len(train_loader)//2:  # 根据验证集loss选择best_net
                #     # print(train_loader.__len__() // 2, j)
                #     self._do_validation(valid_features=valid_features,
                #                         valid_outs=valid_outs,
                #                         increase_better=True,
                #                         seq_mode=seq_mode,
                #                         last=False)

            self._do_validation(valid_features=valid_features,
                                valid_outs=valid_outs,
                                increase_better=True,
                                seq_mode=self.seq_mode,
                                last=False) # 根据验证集loss选择best_net

        self._do_validation(valid_features=valid_features,
                            valid_outs=valid_outs,
                            increase_better=True,
                            seq_mode=self.seq_mode,
                            last=True) # 根据验证集loss选择best_net

    def pred_out_tags(self,
                        ins,
                        tag_i2w,
                        pred_out_tag_path=None,
                        true_outs=None,
                        ):
        logging.info('---Predict the real tags of the sequences')
        if pred_out_tag_path is not None:
            pred_out_tag_dir=os.path.dirname(pred_out_tag_path)
            if not os.path.exists(pred_out_tag_dir):
                os.makedirs(pred_out_tag_dir)
        pred_outs,_=self.predict(ins)
        pred_out_tags=[[tag_i2w[out_idx] for out_idx in pred_out_seq[:list(pred_out_seq).index(0)]]
                         for pred_out_seq in pred_outs]
        if pred_out_tag_path is not None:
            feature_content=[' '.join(feature_seq) for feature_seq in ins]
            pred_out_tag_content = [' '.join(out_tag_seq) for out_tag_seq in pred_out_tags]
            if true_outs is not None:
                true_out_tag_content = [' '.join([tag_i2w[out_idx] for out_idx in true_out_seq])
                                   for true_out_seq in true_outs]
                content='\n\n'.join(['\n'.join(content_tuple) for content_tuple in zip(feature_content,true_out_tag_content,pred_out_tag_content)])
            else:
                content='\n\n'.join(['\n'.join(content_tuple) for content_tuple in zip(feature_content,pred_out_tag_content)])
            with codecs.open(pred_out_tag_path,'w') as f:
                f.write(content)
        return pred_out_tags    #别忘了return

if __name__=='__main__':
    logging.info('Load data ...')
    with codecs.open(avail_train_path, 'rb') as f:
        train_features, train_pos_labels, train_chunk_labels, train_ner_labels = zip(*pickle.load(f))
        # print(train_pos_labels[:10])
        # print(train_ner_labels[:10])
    with codecs.open(avail_dev_path, 'rb') as f:
        dev_features, dev_pos_labels, dev_chunk_labels, dev_ner_labels = zip(*pickle.load(f))
        # print(test_outs[:100])
    with codecs.open(avail_test_path, 'rb') as f:
        test_ins, test_pos_labels, test_chunk_labels, test_ner_labels = zip(*pickle.load(f))

    with codecs.open(pos2idx_path, 'r') as f:
        pos2idx = json.load(f)
        idx2pos = {}
        for key, value in pos2idx.items():
            idx2pos[value] = key
    # print(idx2pos)

    with codecs.open(ner2idx_path, 'r') as f:
        ner2idx = json.load(f)
        idx2ner = {}
        for key, value in ner2idx.items():
            idx2ner[value] = key
    # print(idx2ner)

    with codecs.open(chunk2idx_path, 'r') as f:
        chunk2idx = json.load(f)
        idx2chunk = {}
        for key, value in chunk2idx.items():
            idx2chunk[value] = key

    with codecs.open(train_w2i2w_path, 'rb') as f:
        w2i2w = pickle.load(f)
    out_i2w = w2i2w['out_i2w']

    # print(sorted(list(np.unique(np.concatenate(train_ner_labels)))))  # 排序的unique outs
    # print(sorted(list(np.unique(np.concatenate(dev_ner_labels)))))  # 排序的unique outs
    # print(sorted(list(np.unique(np.concatenate(test_ner_labels)))))  # 排序的unique outs
    #
    # print(sorted(list(np.unique(np.concatenate(train_pos_labels)))))  # 排序的unique outs
    # print(sorted(list(np.unique(np.concatenate(dev_pos_labels)))))  # 排序的unique outs
    # print(sorted(list(np.unique(np.concatenate(test_pos_labels)))))  # 排序的unique outs

    # train_set = Datasetx(train_features, train_labels)
    # train_loader = DataLoader(dataset=train_set,
    #                           batch_size=10,
    #                           shuffle=False)
    # for batch_features,batch_outs in train_loader:
    #     print(batch_features.size())

    # train_metrics=[get_pearson_corr_val]
    # eval_metrics=[get_pearson_corr_val,get_spearman_corr_val]
    # train_metrics = [get_overall_accuracy]
    # valid_metric=get_overall_accuracy
    # eval_metrics = [get_overall_accuracy, get_macro_F1_score]
    model = TransSeqLabel(model_dir=model_dir,
                     model_name='Transformer_based_model',
                     model_id=model_id,
                     embed_dim=embed_dim,
                     token_embed_path=train_token_embed_path,
                     token_embed_freeze=token_embed_freeze,
                     head_num=head_num,
                     att_layer_num=att_layer_num,
                     drop_rate=drop_rate,
                     batch_size=batch_size,
                     big_epochs=big_epochs,
                     regular_rate=regular_rate,  # 1e -5
                     lr_base=lr_base,
                     lr_decay=lr_decay,
                     min_lr_rate=0.01,
                 warm_big_epochs=2,
                 Net=TransNet,
                 Dataset=Datasetx,
                     )  # 初始化模型对象的一个实例
    model.fit(train_features=train_features,
              train_outs=train_ner_labels,
              valid_features=dev_features,
              valid_outs=dev_ner_labels,
              out2tag=idx2ner,
              tag2span_func=tag2span_bio,
              train_metrics=train_metrics,
              valid_metric=valid_metric,
              verbose=verbose)
    # model.save_params(model_params_dir=model_dir)
    # model.load_params(model_params_dir=model_dir)
    # # 预测模型
    # pred_outs = model.predict_outs(test_ins)
    # # 测试混淆矩阵
    # unique_outs = list(np.unique(np.array(train_outs)))  # 不同标记列表
    # confusion_matrix = get_confusion_matrix_df(test_outs, pred_outs,
    #                                            unique_outs=unique_outs)
    # print('confusion matrix is as follows:', confusion_matrix, sep='\n')

    # 用验证集测试模型性能
    # eval_df = model.evaluate(test_ins=eval_features,
    #                          test_outs=eval_labels,
    #                          eval_metrics=eval_metrics
    #                          )
    # print('Model performance on evaluation dataset:\n',eval_df)

    # 用开发集测试模型性能
    model.save_pred_outs(test_ins=test_ins,
                         test_outs=test_ner_labels,
                         pos_test_outs=test_pos_labels,
                         pred_outs_path=pred_ner_path,
                         out_i2w=out_i2w,
                         idx2tag=idx2ner,
                         idx2pos=idx2pos
                         )
    eval_df = model.eval_label(test_ins=test_ins,
                               test_outs=test_ner_labels,
                               eval_metrics=eval_metrics
                               )
    print('Model performance on test dataset:\n', eval_df)