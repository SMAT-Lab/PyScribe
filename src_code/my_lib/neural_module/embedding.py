#coding=utf-8
import os
import logging
import codecs
import numpy as np
import pickle
import json
import torch
import torch.nn as nn

def parse_glove(glove_path,emb_dim=300):
    logging.info('########### Start parsing glove ##########')
    words=['<PAD>','<UNK>']
    embs=[[0]*emb_dim,[1]*emb_dim]
    # emb_arr2=np.zeros((2,emb_dim),dtype=np.float)
    # emb_arr2[1,:]=1.0
    # print(np.array(embs).shape,np.array(embs).dtype)
    with codecs.open(glove_path,'r') as f:
        # print(f.__sizeof__())
        # emb_arr2=np.zeros((len(list(f)),emb_dim),dtype=np.float32)
        # f.seek(0)
        # j=0
        for line in f:
            elements=line.rstrip().split(' ')
            words.append(elements[0])
            # print(np.asarray(elements[1:], dtype=np.float).dtype)
            # print(elements[1:])
            embs.append(elements[1:])
            # if len(elements[1:])==299:
            #     # print(i,elements)
            #     print(i)
            #     j+=1
            #     continue
            # emb_arr2[i,:]=np.asarray(elements[1:],dtype=np.float32)
            # new_emb_arr2=np.asarray(([elements[1:]]),dtype=np.float)
            # emb_arr2=np.concatenate((emb_arr2,new_emb_arr2),axis=0)
            # if np.asarray(elements[1:], dtype=np.float).dtype==object:
            #     print(i,elements[1:])
            # if i>2196000:
            #     print(np.array(embs).shape,np.array(embs).dtype)
    # print('lenght of 299:',j)
    # print(embs[-2:])
    emb_arr2=np.asarray(embs,dtype=np.float32)
    # words=['<PAD>','<UNK>']+words
    # add_arr2=np.asarray([[0]*emb_dim,[1]*emb_dim],dtype=np.float)
    # emb_arr2=np.concatenate((add_arr2,emb_arr2),axis=0)
    # print(emb_arr2.shape,emb_arr2.dtype)
    # print(emb_arr2[-1,:])
    emb_arr2[1,:]=np.random.normal(emb_arr2[2,:].mean(axis=0),emb_arr2.std(axis=0),
                                     size=(emb_dim,))
    glove_dir=os.path.dirname(glove_path)
    np.save(os.path.join(glove_dir,'embed_weight.npy'),emb_arr2)
    word2idx, idx2word = {}, {}  # 初始化word-id和id-word字典
    logging.info('Make the dictionary')
    # idx = 0  # 初始化idx
    for idx, word in enumerate(words):
        # 对word进行数字化编码，保证了<PAD>标号为0，<UNK>编号为1，以便于后续使用
        word2idx[word] = idx
        idx2word[idx] = word
        # idx += 1
    # print(str(idx2word[898]))
    # 打包word-id和id-word字典
    w2i2w = {'word2idx': word2idx, 'idx2word': idx2word}

    # pickle打包
    w2i2w_path=os.path.join(glove_dir,'w2i2w.pkl')
    logging.info('Save the dictionary into %s' % w2i2w_path)
    with codecs.open(w2i2w_path, 'wb') as f:
        pickle.dump(w2i2w, f)
    # 存入json以便观察
    w2i2w_json_path = os.path.splitext(w2i2w_path)[0] + '.json'
    logging.info('Save the dictionary into %s' % w2i2w_json_path)
    with codecs.open(w2i2w_json_path, 'w', encoding='utf-8') as f:
        json.dump(w2i2w, f, indent=4)

    logging.info('########### Finish parsing glove ##########')

class PosEnc(nn.Module):

    def __init__(self,max_len, emb_dim,train=True,pad=True):
        """初始化。

        Args:
            emb_dim: 一个标量。模型的维度，论文默认是512
            max_len: 一个标量。文本序列的最大长度
            train: 是否使用可训练的位置编码
        """
        super().__init__()
        self.pad=pad
        if not train:
            # 根据论文给的公式，构造出PE矩阵
            position_code = np.array([
                [pos / np.power(10000, 2.0 * (j // 2) / emb_dim) for j in range(emb_dim)]
                for pos in range(max_len)])
            # 偶数列使用sin，奇数列使用cos
            position_code[:, 0::2] = np.sin(position_code[:, 0::2])
            position_code[:, 1::2] = np.cos(position_code[:, 1::2])
            position_code=torch.tensor(position_code).float()

            # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
            # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
            # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
            # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
            if pad:
                pad_row = torch.zeros(1, emb_dim)
                # print(torch.tensor(position_code).dtype)
                position_code = torch.cat((pad_row, position_code),dim=0)

                # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
                # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
                self.position_encoder = nn.Embedding(max_len + 1, emb_dim,padding_idx=0)
                self.position_encoder.weight = nn.Parameter(position_code,requires_grad=False)
            else:
                self.position_encoder = nn.Embedding(max_len, emb_dim, padding_idx=None)
                self.position_encoder.weight = nn.Parameter(position_code,requires_grad=False)
        else:
            if pad:
                self.position_encoder=nn.Embedding(max_len+1,emb_dim,padding_idx=0)
                # nn.init.xavier_uniform_(self.position_encoder.weight[1:,:])
            else:
                self.position_encoder = nn.Embedding(max_len, emb_dim, padding_idx=None)
                # nn.init.xavier_uniform_(self.position_encoder.weight)
            # print(self.position_encoder.weight)
            # print(self.position_encoder.weight.data[0,:])
            # print()

    def forward(self, x):
        """神经网络的前向传播。

        Args:
          x: 一个张量，形状为[BATCH_SIZE, max_len]或[BATCH_SIZE, max_len，emb_dim]。每一个张量的值代表这一批文本序列中对应的长度。

        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """

        # 找出这一批序列的最大长度
        # max_len = x.size(1)
        tensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        if self.pad:
            if len(x.size())==3: #x,(B,L,D)
                x_lens = x.abs().sum(2).sign().sum(1).to('cpu').int().data.numpy()  # (B,)   #每个长度
            elif len(x.size())==2:  #x,(B,L)
                # x_len=x.abs().sign().sum(1)
                x_lens=x.abs().sign().sum(1).int().cpu().data.numpy()    #(B,)   #每个长度
                # x_lens=x.abs()
                # pass

            # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
            # 这里range从1开始也是因为要避开PAD(0)的位置
            x_pos = tensor([list(range(1, x_len + 1)) + [0] * (x.size(1) - x_len) for x_len in x_lens]) #(B,L)
        else:   #如果不做pad，就是空字符处的位置向量也算上
            x_pos=tensor(range(x.size(1))).unsqueeze(0).expand(x.size(0),-1)
        return self.position_encoder(x_pos)    #(B,L,D)

class LayerEnc(nn.Module):

    def __init__(self,layer_num, emb_dim,train=False):
        """初始化。

        Args:
            emb_dim: 一个标量。模型的维度，论文默认是512
            layer_num: 一个标量。文本序列的最大长度
            train: 是否使用可训练的位置编码
        """
        super().__init__()
        # self.max_len=max_len
        self.emb_dim=emb_dim
        if not train:
            # 根据论文给的公式，构造出PE矩阵
            layer_code = np.array([
                [pos / np.power(10000, 2.0 * (j // 2) / emb_dim) for j in range(emb_dim)]
                for pos in range(layer_num)])
            # 偶数列使用sin，奇数列使用cos
            layer_code[:, 0::2] = np.sin(layer_code[:, 0::2])
            layer_code[:, 1::2] = np.cos(layer_code[:, 1::2])
            layer_code=torch.tensor(layer_code).float()

            # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
            # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
            self.layer_encoder = nn.Embedding(layer_num, emb_dim,padding_idx=None)
            self.layer_encoder.weight = nn.Parameter(layer_code,requires_grad=False)
        else:
            self.layer_encoder = nn.Embedding(layer_num, emb_dim, padding_idx=None)
            nn.init.xavier_uniform_(self.layer_encoder.weight)
        
    def forward(self,x,i):
        """神经网络的前向传播。

        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """
        # 找出这一批序列的最大长度
        # max_layer_num = x.size(1)
        layer_code=torch.zeros(x.size(0),x.size(1),self.emb_dim,device=x.device)  #(B,L,D)
        i_code=self.layer_encoder(torch.tensor([i],device=x.device))  #(1,D)
        if len(x.size())==3: #x,(B,L,D)
            x_lens = x.sum(2).abs().sign().sum(1).to('cpu').data.numpy().astype(np.int)  # (B,)   #每个长度
        elif len(x.size())==2:  #x,(B,L)
            x_lens=x.abs().sign().sum(1).to('cpu').data.numpy().astype(np.int)   #(B,)   #每个长度
        # tensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        for j,x_len in enumerate(x_lens):
            layer_code[j,:x_len,:]=i_code.expand(x_len,-1)  #对每个序列进行处理
        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        # x_pos = tensor([list(range(1, x_len + 1)) + [0] * (x.size(1) - x_len) for x_len in x_lens])
        return layer_code

# class Embedding(nn.Module):
#
#     def __init__(self, num_embeddings, embedding_dim,scale=False, dropout=0.0, padding_idx=None, sparse=False, max_norm=None, norm_type=2,
#                  scale_grad_by_freq=False,_weight=None):
#         """
#         :param tuple(int,int),torch.FloatTensor,nn.Embedding,numpy.ndarray init_embed: Embedding的大小(传入tuple(int, int),
#             第一个int为vocab_zie, 第二个int为emb_dim); 如果为Tensor, Embedding, ndarray等则直接使用该值初始化Embedding
#         :param None,int padding_idx: 该index的Embedding将一直为0.
#         :param float dropout: 对Embedding的输出的dropout。
#         :param bool sparse: 如果为True，则对Embedding的梯度将是sparse的，参考Pytorch Embedding获取更多信息。
#         :param None,float max_norm: 每个vector最大的norm能为多大
#         :param int norm_type: norm的类型
#         :param bool scale_grad_by_freq: 如果为True，将会把梯度除以这个词出现的次数.
#         """
#         super().__init__()
#         self.num_embeddings=num_embeddings
#         self.embedding_dim=embedding_dim
#         self.scale=scale
#
#         if _weight is not None:
#             assert _weight.shape == (num_embeddings, embedding_dim)  # 断言尺寸
#             self.embedding = nn.Embedding.from_pretrained(_weight, freeze=False)
#         # else:
#         #     super().__init__(num_embeddings=num_embeddings,
#         #                      embedding_dim=embedding_dim,
#         #                      padding_idx=padding_idx,
#         #                      max_norm=max_norm,
#         #                      norm_type=norm_type,
#         #                      scale_grad_by_freq=scale_grad_by_freq,
#         #                      sparse=sparse,
#         #                      _weight=_weight.data)
#         #     self.embedding=super()
#         #     nn.init.xavier_uniform_(embedding.weight)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         """
#         :param torch.LongTensor x: [batch, seq_len]
#         :return: torch.Tensor : [batch, seq_len, emb_dim]
#         """
#         x = self.embedding(x)
#
#         if self.scale:
#             x=x.mul(self.embedding_dim)
#         return self.dropout(x)
#
#     def size(self):
#         """
#         Embedding的大小
#         :return: torch.Size()
#         """
#         return (self.num_embeddings,self.embedding_dim)

if __name__=='__main__':
    pre_embeddings=torch.randn(5,10)
    embed_layer=Embedding(5,10,_weight=pre_embeddings)
    embed=embed_layer(torch.tensor([[2,4]]))
    print(embed)