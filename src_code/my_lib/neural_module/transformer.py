#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaskedLayerNorm(torch.nn.Module):
    """
    See LayerNorm for details.

    Note, however, that unlike LayerNorm this norm includes a batch component.
    """

    def __init__(self, size: int, gamma0: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(1, 1, size) * gamma0)
        self.beta = torch.nn.Parameter(torch.zeros(1, 1, size))
        self.size = size
        self.eps = eps

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        broadcast_mask = mask.unsqueeze(-1).float()
        num_elements = broadcast_mask.sum() * self.size
        mean = (tensor * broadcast_mask).sum() / num_elements
        masked_centered = (tensor - mean) * broadcast_mask
        std = torch.sqrt(
            (masked_centered * masked_centered).sum() / num_elements + self.eps
        )
        return self.gamma * (tensor - mean) / (std + self.eps) + self.beta

class Tranformer(nn.Module):
    def __init__(self,
                 query_dim=512,
                 key_dim=None,
                 head_num=8,
                 head_dim=None,
                 layer_num=6,
                 drop_rate=0.,
                 causality=False,
                 ):
        '''
        标准transformer
        :param unit_num: int，输入最后对维度（B,L,D)中的D
        :param head_num: int，head的数量，要能被unit_num整除
        :param head_mode: int，head模式（0，1，2）
        :param layer_num: int，层数
        :param drop_rate: float，drop rate
        :param residual: Boolean. 是否加入残差
        :param norm: Boolean. 是否归一化
        '''
        super().__init__()
        self.query_dim = query_dim
        self.key_dim=query_dim if key_dim is None else key_dim
        self.head_num = head_num
        self.head_dim = query_dim // head_num if head_dim is None else head_dim
        # self.head_mode=head_mode
        self.layer_num = layer_num
        self.drop_rate = drop_rate
        # self.layer_norm = nn.LayerNorm(unit_num)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_rate)
        self.attentions = nn.ModuleList(
            [MultiHeadAttention(query_dim=self.query_dim,
                                key_dim=self.key_dim,
                                head_num=self.head_num,
                                head_dim=self.head_dim,
                                drop_rate=drop_rate,
                                causality=causality) for _ in range(layer_num)]
        )
        self.layer_norms1 = nn.ModuleList(
            [nn.LayerNorm(self.query_dim, elementwise_affine=True) for _ in range(layer_num)]
        )
        self.forwards = nn.ModuleList(
            [FeedForward(in_dim=self.query_dim,
                         hid_dim=self.query_dim * 4,
                         drop_rate=drop_rate) for _ in range(layer_num)]
        )
        self.layer_norms2 = nn.ModuleList(
            [nn.LayerNorm(self.query_dim, elementwise_affine=True) for _ in range(layer_num)]
        )

    def forward(self, query,key,query_mask=None,key_mask=None):
        '''

        :param x: 3-D shaped tensor, (B,L,D)
        :return: [batch, length, hidden] the output sequence
                 [batch, hidden] the global relay node
        '''
        # x_len=x.size(1) #L
        if query_mask is None:
            query_len = query.size(1)  # L
            batch_max_query_len = query_mask.sum(1).max().int()  # L-,实际最大长度（去0的）
            query = query[:, :batch_max_query_len, :]  # (B,L-,D)
            query_mask = query_mask[:, :batch_max_query_len]  # (B,L-)
        if key_mask is None:
            # key_len = query.size(1)  # L
            batch_max_key_len = key_mask.sum(1).max().int()  # L-,实际最大长度（去0的）
            query = query[:, :batch_max_key_len, :]  # (B,L-,D)
            key_mask = key_mask[:, :batch_max_key_len]  # (B,L-)
        for i in range(self.layer_num):
            query_ = self.attentions[i](query=query, key=key,query_mask=query_mask,key_mask=key_mask)  # (B,L-,D)
            query=self.layer_norms1[i](query_.add(query))
            query_=self.forwards[i](query,mask=query_mask)   # (B,L-,D)
            query=self.layer_norms2[i](query_.add(query))
        if query_mask is None:
            query=query.mul(query.unsqueeze(-1).expand(-1, -1, self.unit_num).float())
            query=F.pad(query,(0,0,0,query_len-batch_max_query_len,0,0))  #padding，to (B,L,D)
        return query  # (B,L,D)

class TranEnc(nn.Module):
    def __init__(self,
                 query_dim=512,
                 head_num=8,
                 head_dim=None,
                 ff_hid_dim=2048,
                 layer_num=6,
                 drop_rate=0.,
                 ):
        '''
        标准transformer
        :param unit_num: int，输入最后对维度（B,L,D)中的D
        :param head_num: int，head的数量，要能被unit_num整除
        :param head_mode: int，head模式（0，1，2）
        :param layer_num: int，层数
        :param drop_rate: float，drop rate
        :param residual: Boolean. 是否加入残差
        :param norm: Boolean. 是否归一化
        '''
        super().__init__()
        # print(layer_num)
        self.query_dim = query_dim
        self.head_num = head_num
        self.layer_num=layer_num
        self.head_dim = query_dim // head_num if head_dim is None else head_dim
        # self.layer_num = layer_num
        self.enc_blocks = nn.ModuleList([EncBlock(query_dim=self.query_dim,
                                                  # key_dim=self.key_dim,
                                                  head_num=self.head_num,
                                                  head_dim=self.head_dim,
                                                  ff_hid_dim=ff_hid_dim,
                                                  drop_rate=drop_rate,) for _ in range(layer_num)])

    def forward(self, query,query_mask=None):
        '''

        :param x: 3-D shaped tensor, (B,L,D)
        :return: [batch, length, hidden] the output sequence
                 [batch, hidden] the global relay node
        '''
        # x_len=x.size(1) #L
        flag=0
        if query_mask is None:
            flag=1
            query_len = query.size(1)  # L
            batch_max_query_len = query.abs().sum(-1).sign().sum(-1).max().int()  # L-,实际最大长度（去0的）
            query = query[:, :batch_max_query_len, :]  # (B,L-,D)
            query_mask = query.abs().sum(-1).sign()  # (B,L-)
        for i in range(self.layer_num):
            query=self.enc_blocks[i](query=query,query_mask=query_mask)
        if flag==1:
            query=query.mul(query_mask.unsqueeze(-1).expand(-1, -1, self.query_dim).float())
            query=F.pad(query,[0,0,0,query_len-batch_max_query_len,0,0])  #padding，to (B,L,D)
        return query  # (B,L,D)

class DualTranDec(nn.Module):
    def __init__(self,
                 query_dim=512,
                 key_dim=None,
                 head_num=8,
                 ff_hid_dim=2048,
                 head_dim=None,
                 layer_num=6,
                 drop_rate=0.,
                 mode='sequential'
                 ):
        '''

        :param query_dim:
        :param key_dim:
        :param head_num:
        :param ff_hid_dim:
        :param head_dim:
        :param layer_num:
        :param drop_rate:
        :param mode: 'sequential'means attention over key1 first and key2 second,
                        'parallel' means attention over key1 and key2 in the same time.
        '''
        super().__init__()
        self.query_dim = query_dim
        self.key_dim=query_dim if key_dim is None else key_dim
        self.head_num = head_num
        self.head_dim = query_dim // head_num if head_dim is None else head_dim
        self.layer_num = layer_num
        # self.mode=mode
        self.dec_blocks = nn.ModuleList([DualDecBlock(query_dim=self.query_dim,
                                                      key_dim=self.key_dim,
                                                      head_num=self.head_num,
                                                      head_dim=self.head_dim,
                                                      ff_hid_dim=ff_hid_dim,
                                                      drop_rate=drop_rate,
                                                      mode=mode) for _ in range(layer_num)])

    def forward(self, query,key1,key2,query_mask=None,key_mask1=None,key_mask2=None):
        '''

        :param x: 3-D shaped tensor, (B,L,D)
        :return: [batch, length, hidden] the output sequence
                 [batch, hidden] the global relay node
        '''
        # x_len=x.size(1) #L
        flag=0
        if query_mask is None:
            flag=1
            query_len = query.size(1)  # L
            batch_max_query_len = query.abs().sum(-1).sign().sum(-1).max().int()  # L-,实际最大长度（去0的）
            query = query[:, :batch_max_query_len, :]  # (B,L-,D)
            query_mask = query.abs().sum(-1).sign()  # (B,L-)
        if key_mask1 is None:
            # key_len = query.size(1)  # L
            batch_max_key_len = key1.abs().sum(-1).sign().sum(-1).max().int()  # L-,实际最大长度（去0的）
            key1= key1[:, :batch_max_key_len, :]  # (B,L-,D)
            key_mask1 = key1.abs().sum(-1).sign()  # (B,L-)
        if key_mask2 is None:
            # key_len = query.size(1)  # L
            batch_max_key_len = key2.abs().sum(-1).sign().sum(-1).max().int()  # L-,实际最大长度（去0的）
            key2= key2[:, :batch_max_key_len, :]  # (B,L-,D)
            key_mask2 = key2.abs().sum(-1).sign()  # (B,L-)
        for i in range(self.layer_num):
            query=self.dec_blocks[i](query=query,
                                     key1=key1,
                                     key2=key2,
                                     query_mask=query_mask,
                                     key_mask1=key_mask1,
                                     key_mask2=key_mask2,
                                     )
        if flag==1:
            query=query.mul(query_mask.unsqueeze(-1).expand(-1, -1, self.query_dim).float())
            query=F.pad(query,[0,0,0,query_len-batch_max_query_len,0,0])  #padding，to (B,L,D)
        return query  # (B,L,D)

class TranDec(nn.Module):
    def __init__(self,
                 query_dim=512,
                 key_dim=None,
                 head_num=8,
                 ff_hid_dim=2048,
                 head_dim=None,
                 layer_num=6,
                 drop_rate=0.,
                 ):
        '''
        标准transformer
        :param unit_num: int，输入最后对维度（B,L,D)中的D
        :param head_num: int，head的数量，要能被unit_num整除
        :param head_mode: int，head模式（0，1，2）
        :param layer_num: int，层数
        :param drop_rate: float，drop rate
        :param residual: Boolean. 是否加入残差
        :param norm: Boolean. 是否归一化
        '''
        super().__init__()
        self.query_dim = query_dim
        self.key_dim=query_dim if key_dim is None else key_dim
        self.head_num = head_num
        self.head_dim = query_dim // head_num if head_dim is None else head_dim
        self.layer_num = layer_num
        self.dec_blocks = nn.ModuleList([DecBlock(query_dim=self.query_dim,
                                 key_dim=self.key_dim,
                                 head_num=self.head_num,
                                 head_dim=self.head_dim,
                                    ff_hid_dim=ff_hid_dim,
                                 drop_rate=drop_rate,) for _ in range(layer_num)])

    def forward(self, query,key,query_mask=None,key_mask=None):
        '''

        :param x: 3-D shaped tensor, (B,L,D)
        :return: [batch, length, hidden] the output sequence
                 [batch, hidden] the global relay node
        '''
        # x_len=x.size(1) #L
        flag=0
        if query_mask is None:
            flag=1
            query_len = query.size(1)  # L
            batch_max_query_len = query.abs().sum(-1).sign().sum(-1).max().int()  # L-,实际最大长度（去0的）
            query = query[:, :batch_max_query_len, :]  # (B,L-,D)
            query_mask = query.abs().sum(-1).sign()  # (B,L-)
        if key_mask is None:
            # key_len = query.size(1)  # L
            batch_max_key_len = key.abs().sum(-1).sign().sum(-1).max().int()  # L-,实际最大长度（去0的）
            key= key[:, :batch_max_key_len, :]  # (B,L-,D)
            key_mask = key.abs().sum(-1).sign()  # (B,L-)
        for i in range(self.layer_num):
            query=self.dec_blocks[i](query=query,key=key,query_mask=query_mask,key_mask=key_mask)
        if flag==1:
            query=query.mul(query_mask.unsqueeze(-1).expand(-1, -1, self.query_dim).float())
            query=F.pad(query,[0,0,0,query_len-batch_max_query_len,0,0])  #padding，to (B,L,D)
        return query  # (B,L,D)

class EncBlock(nn.Module):
    def __init__(self,
                 query_dim=512,
                 head_num=8,
                 head_dim=None,
                 ff_hid_dim=2048,
                 drop_rate=0.,
                 # causality=False,
                 ):
        '''
        带残差的Multi-head Attention
        :param unit_num: int，输入最后对维度（B,L,D)中的D
        :param head_num: int，head的数量，要能被unit_num整除
        :param head_mode: int，head模式（0，1，2）
        :param layer_num: int，层数
        :param drop_rate: float，drop rate
        :param residual: Boolean. 是否加入残差
        :param norm: Boolean. 是否归一化
        '''
        super().__init__()
        self.query_dim = query_dim
        self.head_num = head_num
        self.head_dim = query_dim // head_num if head_dim is None else head_dim
        self.res_att=ResMHA(query_dim=query_dim,
                            key_dim=query_dim,
                            head_num=head_num,
                            head_dim=head_dim,
                            drop_rate=drop_rate,
                            causality=False, )
        self.res_ff = ResFF(in_dim=query_dim,
                            hid_dim=ff_hid_dim,
                            # out_dim=query_dim,
                            drop_rate=drop_rate)
        # self.res_lstm = ResLSTM(in_dim=query_dim,
        #                         hid_dim=query_dim * 2,
        #                         drop_rate=drop_rate)
    def forward(self, query,query_mask):
        '''

        :param x: 3-D shaped tensor, (B,L,D)
        :return: [batch, length, hidden] the output sequence
                 [batch, hidden] the global relay node
        '''
        query=self.res_att(query=query, key=query,query_mask=query_mask,key_mask=query_mask)
        query=self.res_ff(query,mask=query_mask)
        # query=self.res_lstm(query,mask=query_mask)
        return query  # (B,L,D)

class DualDecBlock(nn.Module):
    def __init__(self,
                 query_dim=512,
                 key_dim=None,
                 ff_hid_dim=2048,
                 head_num=8,
                 head_dim=None,
                 drop_rate=0.,
                 mode='sequential'
                 ):
        '''

        :param query_dim:
        :param key_dim:
        :param ff_hid_dim:
        :param head_num:
        :param head_dim:
        :param drop_rate:
        :param mode: 'sequential'means attention over key1 first and key2 second,
                        'parallel' means attention over key1 and key2 in the same time.
        '''
        super().__init__()
        self.query_dim = query_dim
        self.key_dim=query_dim if key_dim is None else key_dim
        self.head_num = head_num
        self.head_dim = query_dim // head_num if head_dim is None else head_dim
        self.mode=mode
        self.res_self_att=ResMHA(query_dim=query_dim,
                                 key_dim=query_dim,
                                 head_num=head_num,
                                 head_dim=head_dim,
                                 drop_rate=drop_rate,
                                 causality=True)
        self.res_cross_att1 = ResMHA(query_dim=query_dim,
                                    key_dim=key_dim,
                                    head_num=head_num,
                                    head_dim=head_dim,
                                    drop_rate=drop_rate,
                                    causality=False)
        self.res_cross_att2 = ResMHA(query_dim=query_dim,
                                    key_dim=key_dim,
                                    head_num=head_num,
                                    head_dim=head_dim,
                                    drop_rate=drop_rate,
                                    causality=False)
        self.res_ff= ResFF(in_dim=query_dim,
                                    hid_dim=ff_hid_dim,
                                    # out_dim=query_dim,
                                    drop_rate=drop_rate)
        # self.res_lstm = ResLSTM(in_dim=query_dim,
        #                     hid_dim=query_dim * 2,
        #                     drop_rate=drop_rate)
    def forward(self, query,key1,key2,query_mask,key_mask1,key_mask2):
        '''

        :param x: 3-D shaped tensor, (B,L,D)
        :return: [batch, length, hidden] the output sequence
                 [batch, hidden] the global relay node
        '''
        query=self.res_self_att(query=query, key=query,query_mask=query_mask,key_mask=query_mask)
        # query = self.res_lstm(query, mask=query_mask)
        if self.mode=='sequential':
            query=self.res_cross_att1(query=query,key=key1,query_mask=query_mask,key_mask=key_mask1)
            query=self.res_cross_att2(query=query,key=key2,query_mask=query_mask,key_mask=key_mask2)
        if self.mode=='parallel':
            query1 = self.res_cross_att1(query=query, key=key1, query_mask=query_mask, key_mask=key_mask1)
            query2 = self.res_cross_att2(query=query, key=key2, query_mask=query_mask, key_mask=key_mask2)
            query=query1.add(query2)
        query=self.res_ff(query,mask=query_mask)
        # query=self.res_lstm(query,mask=query_mask)
        return query  # (B,L,D)



class DecBlock(nn.Module):
    def __init__(self,
                 query_dim=512,
                 key_dim=None,
                 ff_hid_dim=2048,
                 head_num=8,
                 head_dim=None,
                 drop_rate=0.
                 ):
        '''
        :param unit_num: int，输入最后对维度（B,L,D)中的D
        :param head_num: int，head的数量，要能被unit_num整除
        :param head_mode: int，head模式（0，1，2）
        :param layer_num: int，层数
        :param drop_rate: float，drop rate
        :param residual: Boolean. 是否加入残差
        :param norm: Boolean. 是否归一化
        '''
        super().__init__()
        self.query_dim = query_dim
        self.key_dim=query_dim if key_dim is None else key_dim
        self.head_num = head_num
        self.head_dim = query_dim // head_num if head_dim is None else head_dim
        self.res_self_att=ResMHA(query_dim=query_dim,
                                 key_dim=query_dim,
                                 head_num=head_num,
                                 head_dim=head_dim,
                                 drop_rate=drop_rate,
                                 causality=True)
        self.res_cross_att = ResMHA(query_dim=query_dim,
                                    key_dim=key_dim,
                                    head_num=head_num,
                                    head_dim=head_dim,
                                    drop_rate=drop_rate,
                                    causality=False)
        self.res_ff= ResFF(in_dim=query_dim,
                                    hid_dim=ff_hid_dim,
                                    # out_dim=query_dim,
                                    drop_rate=drop_rate)
        # self.res_lstm = ResLSTM(in_dim=query_dim,
        #                     hid_dim=query_dim * 2,
        #                     drop_rate=drop_rate)
    def forward(self, query,key,query_mask,key_mask):
        '''

        :param x: 3-D shaped tensor, (B,L,D)
        :return: [batch, length, hidden] the output sequence
                 [batch, hidden] the global relay node
        '''
        query=self.res_self_att(query=query, key=query,query_mask=query_mask,key_mask=query_mask)
        # query = self.res_lstm(query, mask=query_mask)
        query=self.res_cross_att(query=query,key=key,query_mask=query_mask,key_mask=key_mask)
        query=self.res_ff(query,mask=query_mask)
        # query=self.res_lstm(query,mask=query_mask)
        return query  # (B,L,D)

class ResLSTM(nn.Module):
    def __init__(self,
                 in_dim=512,
                 hid_dim=2048,
                 drop_rate=0.):
        super().__init__()
        self.lstm=nn.LSTM(input_size=in_dim,hidden_size=hid_dim,num_layers=1)
        self.linear=nn.Linear(hid_dim,in_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.layer_norm = nn.LayerNorm(in_dim, elementwise_affine=True)
    def forward(self,x,mask=None):
        self.lstm.flatten_parameters()
        x_,_ = self.lstm(x.transpose(0,1))  # (B,L-,D)
        x_=self.linear(x_.transpose(0,1))
        x = self.layer_norm(x.add(x_))
        x = self.dropout(x)
        # if mask is not None:
        #     mask = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))  # (B, L,D)
        #     x=x.mul(mask.float()) # (B, L,D)
        return x


class ResMHA(nn.Module):
    def __init__(self,
                 query_dim=512,
                 key_dim=None,
                 head_num=8,
                 head_dim=None,
                 drop_rate=0.,
                 causality=False,
                 ):
        '''
        带残差的Multi-head Attention
        :param unit_num: int，输入最后对维度（B,L,D)中的D
        :param head_num: int，head的数量，要能被unit_num整除
        :param head_mode: int，head模式（0，1，2）
        :param layer_num: int，层数
        :param drop_rate: float，drop rate
        :param residual: Boolean. 是否加入残差
        :param norm: Boolean. 是否归一化
        '''
        super().__init__()
        self.query_dim = query_dim
        self.key_dim=query_dim if key_dim is None else key_dim
        self.head_num = head_num
        self.head_dim = query_dim // head_num if head_dim is None else head_dim
        self.attention=MultiHeadAttention(query_dim=self.query_dim,
                                key_dim=self.key_dim,
                                head_num=self.head_num,
                                head_dim=self.head_dim,
                                drop_rate=drop_rate,
                                causality=causality)
        self.layer_norm =nn.LayerNorm(self.query_dim, elementwise_affine=True)


    def forward(self, query,key,query_mask,key_mask):
        '''

        :param x: 3-D shaped tensor, (B,L,D)
        :return: [batch, length, hidden] the output sequence
                 [batch, hidden] the global relay node
        '''
        # x_len=x.size(1) #L
        query_ = self.attention(query=query, key=key,query_mask=query_mask,key_mask=key_mask)  # (B,L-,D)
        query=self.layer_norm(query_.add(query))
        # if query_mask is not None:
        #     query_mask = query_mask.unsqueeze(-1).repeat(1, 1, query.size(-1))  # (B, L,D)
        #     query=query.mul(query_mask.float()) # (B, L,D)
        return query  # (B,L,D)

class ResFF(nn.Module):
    def __init__(self,
                 in_dim=512,
                 hid_dim=2048,
                 drop_rate=0.):
        super().__init__()
        self.feedforward = FeedForward(in_dim=in_dim,
                                    hid_dim=hid_dim,
                                    out_dim=in_dim,
                                    drop_rate=drop_rate)
        self.layer_norm = nn.LayerNorm(in_dim, elementwise_affine=True)
        # print('feed forward')
    def forward(self, x,mask=None):
        x_ = self.feedforward(x, mask=mask)  # (B,L-,D)
        x = self.layer_norm(x.add(x_))
        # if mask is not None:
        #     mask = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))  # (B, L,D)
        #     x=x.mul(mask.float()) # (B, L,D)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 query_dim=512,
                 key_dim=None,
                 head_num=8,
                 head_dim=None,
                 drop_rate=0.,
                 # residual=True,
                 # norm=True,
                 causality=False,
                 ):
        '''
        init
        :param unit_num: A scalar. Attention size.
        :param head_num: An int. Number of heads.
        :param drop_rate: A floating point number.
        :param head_mode: An int. head模式，0表示标准transformer头，1表示共享Linear层transformer头，2表示共享卷积层transformer头
        :param residual: Boolean. 是否加入残差
        :param norm: Boolean. 是否归一化
        :param causality: Boolean. If true, units that reference the future are masked.
        '''
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = query_dim if key_dim is None else key_dim
        self.head_num=head_num
        self.head_dim = query_dim // head_num if head_dim is None else head_dim
        self.hid_dim=self.head_num*self.head_dim
        # self.head_mode=head_mode
        # self.residual=residual
        # self.norm=norm

        self.causality=causality
        # io_dims=[self.query_dim,self.key_dim,self.key_dim]
        self.conv1d_ins=nn.ModuleList([nn.Conv1d(io_dim, self.hid_dim, kernel_size=1,padding=0)
                                    for io_dim in [self.query_dim,self.key_dim,self.key_dim]])

        self.conv1d_out=nn.Conv1d(self.hid_dim,self.query_dim, kernel_size=1,padding=0)

        # self.relu=nn.ReLU()
        # self.leaky_relu = nn.LeakyReLU()
        self.softmax=nn.Softmax(dim=-1)
        # self.sigmoid=nn.Sigmoid()
        self.dropout=nn.Dropout(drop_rate)

    def forward(self, query,key,query_mask,key_mask,value=None):
        '''
        Applies multihead attention
        :param query: A 3d tensor with shape of [B, T_q, D_q]
        :param key: A 3d tensor with shape of [B, T_k, D_k]
        :param query_mask: A 3d tensor with shape of [B, T_q]
        :param key_mask: A 3d tensor with shape of [B, T_k]
        :return: A 3d tensor with shape of [B, T_q, D]
        '''
        if value is None:
            value=key.clone()   #深度拷贝
        batch_size=query.size(0)    #B

        #Linear
        query_,key_,value_=[conv1d_in(x.transpose(1,2)) for conv1d_in,x in
                            zip(self.conv1d_ins,(query,key,value))]  #(B,D,L_q),(B,D,L_k),(B,D,L_v)
        query_, key_, value_ = [x.view(batch_size, self.head_num, self.head_dim, -1).transpose(2, 3)
                                    for x in (query_, key_, value_)]  # (B,h,L_q,D/h)(B,h,L_k,D/h)(B,h,L_v,D/h)

        #Multiplication
        # attention=torch.bmm(query_,key_.transpose(1,2).contiguous())   #(B*h,L_q,L_k)
        query_=query_.mul(float(self.head_dim)**-0.5)        #Scale
        attention=torch.einsum('abcd,abed->abce',query_,key_)  #(B,h,L_q,L_k)
        #
        # if self.head_mode!=0:
        #     attention = self.relu(attention)  # (B*h,L_q,L_k)

        #Scale
        # attention=attention / (float(self.head_dim)**0.5)  #(B,h,L_q,L_k)

        #Key Mask
        # if key_mask is None:
        #     key_mask=key.abs().sum(dim=-1)   #(B,L_k)
        if key_mask is not None:
            # value_mask = key_mask[:, None, :, None].expand(-1, self.head_num, -1, self.head_dim)  # (B,h,L_v,D/h)
            key_mask=key_mask.eq(0) #(B,L_k)
            key_mask=key_mask.unsqueeze(dim=1).repeat(1,self.head_num,1)   #(B,h,L_k)
            key_mask=key_mask.unsqueeze(dim=2).expand(-1,-1,query.size(1),-1)    #(B,h,L_q,L_k)
            attention=attention.masked_fill(key_mask,-np.inf)   #(B,h,L_q,L_k)

        # if self.head_mode!=0:
        #     attention=attention.masked_fill(attention<=0,-np.inf)    #(B*h,L_q,L_k)

        # Causality = Future blinding
        if self.causality:
            seq_mask=torch.triu(torch.ones_like(attention[0,0,:,:]),diagonal=1).float() #(L_q,L_k)
            # print(seq_mask)
            # print(seq_mask.size())
            # test_mat=torch.ones_like(attention[0,0,:,:])
            # test_mat=test_mat.masked_fill(seq_mask,-np.inf)
            # print(test_mat)
            seq_mask = seq_mask.masked_fill(seq_mask == 1, float('-inf'))
            seq_mask=seq_mask[None,None,:,:].expand(batch_size,self.head_num,-1,-1) #(B,h,L_q,L_k)
            # print('seq_mask:',seq_mask.size())
            # print('attention:',attention.size())

            attention=attention.add(seq_mask)
            # attention=attention.masked_fill(seq_mask,float('-inf'))   #(B,h,L_q,L_k)
            # print(attention.size())
            # print(attention[0,0,:,:])
            # print()

        #Softmax

        attention = self.softmax(attention)  # (B,h,L_q,L_k)

        # if query_mask is not None:
        #     query_mask=query_mask[:,None,:].repeat(1,self.head_num,1)  #(B,h,L_q)
        #     query_mask=query_mask.unsqueeze(-1).expand(-1,-1,-1,value.size(1))   #(B,h,L_q,L_k)
        #
        #     # attention*=query_mask
        #     attention=attention.mul(query_mask.float()) #(B,h,L_q,L_k)

        #Dropouts
        attention=self.dropout(attention)   #(B,h,L_q,L_k)
        # if query.size(-1)>1:
        # print('attention2:',attention[:,-3:,:])

        # if key_mask is not None:
        #     # key_mask=key_mask[:,None,:,None].expand(-1,self.head_num,-1,self.head_dim)  #(B,h,L_v,D/h)
        #
        #     # attention*=query_mask
        #     value_=value_.mul(value_mask.float()) #(B,h,L_v,D/h)

        # Weighted sum
        output=torch.matmul(attention,value_)  #(B,h,L_q,D/h)
        output=output.transpose(1,2).contiguous().view(batch_size,-1,self.hid_dim)    #(B,L_q,D)
        output=self.conv1d_out(output.transpose(1,2)).transpose(1,2)  #(B,L_q,D)

        if query_mask is not None:
            query_mask=query_mask[:,:,None].repeat(1,1,self.query_dim)  #(B,L_q,D)
            output=output.mul(query_mask.float()) #(B,L_q,D)
        # # print(output)
        # if self.residual:
        #     # Residual connection
        #     output=output.add(query)   #(B,L_q,D)
        #
        #     if self.head_mode!=0:
        #         output = self.relu(output)  # (B,L_q,D)
        #
        # if self.norm:
        #     # Normalize
        #     output=self.layer_norm(output)  #(B,L_q,D)
        #
        # # print('query_mask:',query.abs().sum(dim=-1).sign())   #(B,L_q)
        # # print('output_mask:',output.abs().sum(dim=-1).sign())   #(B,L_q)

        return output

class FeedForward(nn.Module):
    def __init__(self,
                 in_dim=512,
                 hid_dim=2048,
                 out_dim=None,
                 drop_rate=0.
                 ):
        '''
        :param unit_num: input and output dim
        :param hidden_dim: hidden dim
        :param drop_rate: A floating point number.
        :param residual: Boolean. 是否加入残差
        :param norm: Boolean. 是否归一化
        '''
        super().__init__()
        # self.residual = residual
        # self.norm = norm
        out_dim=in_dim if out_dim is None else out_dim
        self.conv1d_in=nn.Conv1d(in_dim,hid_dim,kernel_size=1,padding=0)
        self.relu=nn.ReLU()
        # self.gelu=nn.GELU()
        self.leaky_relu=nn.LeakyReLU()
        self.conv1d_out=nn.Conv1d(hid_dim,out_dim,kernel_size=1)
        self.dropout = nn.Dropout(drop_rate)
        # self.layer_norm = nn.LayerNorm(unit_num)
        # self.layer_norm = nn.LayerNorm(unit_num)
        # print('feed forward')

    def forward(self, x,mask=None):
        '''
        Point-wise feed forward net
        :param x: A 3d tensor with shape of [B, L, D].
        :param mask: A 3d tensor with shape of [B, L].
        :return: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        '''
        # print('run feed forward')
        output = self.conv1d_in(x.transpose(1, 2))  # (B,D,L)
        output = self.relu(output)  # (B,D,L)
        output = self.conv1d_out(output).transpose(1, 2)  # (B,L,D)
        # if mask is None:
        #     mask = x.abs().sum(dim=-1).sign().float()  # (B,L)
        if mask is not None:
            mask = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))  # (B, L,D)
            output=output.mul(mask.float()) # (B, L,D)

        output=self.dropout(output) #(B,D,L)

        # # output=self.relu(output)
        # if self.residual:
        #     # Residual connection
        #     output=output.add(x)   #(B,L,D)
        # if self.norm:
        #     # Normalize
        #     output=self.layer_norm(output)  #(B,L,D)
        #     # output = self.leaky_relu(output)  # (B,L_q,D)
        return output


# class Transformer(nn.Module):
#     def __init__(self,
#                  unit_num=512,
#                  head_num=8,
#                  head_dim=None,
#                  layer_num=6,
#                  drop_rate=0.,
#                  causality=False,
#                  ):
#         '''
#         标准transformer
#         :param unit_num: int，输入最后对维度（B,L,D)中的D
#         :param head_num: int，head的数量，要能被unit_num整除
#         :param head_mode: int，head模式（0，1，2）
#         :param layer_num: int，层数
#         :param drop_rate: float，drop rate
#         :param residual: Boolean. 是否加入残差
#         :param norm: Boolean. 是否归一化
#         '''
#         super().__init__()
#         self.unit_num = unit_num
#         self.head_num = head_num
#         self.head_dim = head_dim
#         if head_dim is None:
#             self.head_dim=self.unit_num//self.head_num
#         # self.head_mode=head_mode
#         self.layer_num = layer_num
#         self.drop_rate = drop_rate
#         # self.layer_norm = nn.LayerNorm(unit_num)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=drop_rate)
#         self.attentions = nn.ModuleList(
#             [MultiHeadAttention(unit_num=unit_num,
#                                 head_num=head_num,
#                                 drop_rate=drop_rate,
#                                 head_dim=self.head_dim,
#                                 causality=causality) for _ in range(layer_num)]
#         )
#         self.layer_norms1 = nn.ModuleList(
#             [nn.LayerNorm(unit_num, elementwise_affine=True) for _ in range(layer_num)]
#         )
#         self.forwards = nn.ModuleList(
#             [FeedForward(unit_num=unit_num,
#                          hidden_dim=unit_num * 4,
#                          drop_rate=drop_rate) for _ in range(layer_num)]
#         )
#         self.layer_norms2 = nn.ModuleList(
#             [nn.LayerNorm(unit_num, elementwise_affine=True) for _ in range(layer_num)]
#         )
#
#     def forward(self, query,key,query_mask=None,key_mask=None):
#         '''
#
#         :param x: 3-D shaped tensor, (B,L,D)
#         :return: [batch, length, hidden] the output sequence
#                  [batch, hidden] the global relay node
#         '''
#         # x_len=x.size(1) #L
#         if query_mask is None:
#             query_len = query.size(1)  # L
#             batch_max_query_len = query_mask.sum(1).max().int()  # L-,实际最大长度（去0的）
#             query = query[:, :batch_max_query_len, :]  # (B,L-,D)
#             query_mask = query_mask[:, :batch_max_query_len]  # (B,L-)
#         if key_mask is None:
#             # key_len = query.size(1)  # L
#             batch_max_key_len = key_mask.sum(1).max().int()  # L-,实际最大长度（去0的）
#             query = query[:, :batch_max_key_len, :]  # (B,L-,D)
#             key_mask = key_mask[:, :batch_max_key_len]  # (B,L-)
#         for i in range(self.layer_num):
#             query_ = self.attentions[i](query=query, key=key,query_mask=query_mask,key_mask=key_mask)  # (B,L-,D)
#             query=self.layer_norms1[i](query_.add(query))
#             query_=self.forwards[i](query,mask=query_mask)   # (B,L-,D)
#             query=self.layer_norms2[i](query_.add(query))
#         if query_mask is None:
#             query=query.mul(query.unsqueeze(-1).expand(-1, -1, self.unit_num).float())
#             query=F.pad(query,(0,0,0,query_len-batch_max_query_len,0,0))  #padding，to (B,L,D)
#         return query  # (B,L,D)
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self,
#                  unit_num=512,
#                  head_num=8,
#                  head_dim=None,
#                  drop_rate=0.,
#                  # residual=True,
#                  # norm=True,
#                  causality=False,
#                  ):
#         '''
#         init
#         :param unit_num: A scalar. Attention size.
#         :param head_num: An int. Number of heads.
#         :param drop_rate: A floating point number.
#         :param head_mode: An int. head模式，0表示标准transformer头，1表示共享Linear层transformer头，2表示共享卷积层transformer头
#         :param residual: Boolean. 是否加入残差
#         :param norm: Boolean. 是否归一化
#         :param causality: Boolean. If true, units that reference the future are masked.
#         '''
#         super().__init__()
#         self.head_num=head_num
#         self.head_dim = head_dim
#         if head_dim is None:
#             self.head_dim = unit_num // head_num
#         self.hidden_dim=self.head_num*self.head_dim
#         # self.head_mode=head_mode
#         # self.residual=residual
#         # self.norm=norm
#         self.unit_num=unit_num
#         self.causality=causality
#         self.conv1ds=nn.ModuleList([nn.Conv1d(unit_num, self.hidden_dim, kernel_size=1,padding=0) for _ in range(3)])
#
#         self.conv1d_o=nn.Conv1d(self.hidden_dim,unit_num, kernel_size=1,padding=0)
#
#         self.relu=nn.ReLU()
#         self.leaky_relu = nn.LeakyReLU()
#         self.softmax=nn.Softmax(dim=-1)
#         self.sigmoid=nn.Sigmoid()
#         self.dropout=nn.Dropout(drop_rate)
#
#     def forward(self, query,key,query_mask,key_mask,value=None):
#         '''
#         Applies multihead attention
#         :param query: A 3d tensor with shape of [B, T_q, D_q]
#         :param key: A 3d tensor with shape of [B, T_k, D_k]
#         :param query_mask: A 3d tensor with shape of [B, T_q]
#         :param key_mask: A 3d tensor with shape of [B, T_k]
#         :return: A 3d tensor with shape of [B, T_q, D]
#         '''
#         if value is None:
#             value=key.clone()   #深度拷贝
#         batch_size=query.size(0)    #B
#
#         #Linear
#         query_,key_,value_=[conv1d(x.transpose(1,2)) for conv1d,x in
#                             zip(self.conv1ds,(query,key,value))]  #(B,D,L_q),(B,D,L_k),(B,D,L_v)
#         query_, key_, value_ = [x.view(batch_size, self.head_num, self.head_dim, -1).transpose(2, 3)
#                                     for x in (query_, key_, value_)]  # (B,h,L_q,D/h)(B,h,L_k,D/h)(B,h,L_v,D/h)
#
#         #Multiplication
#         # attention=torch.bmm(query_,key_.transpose(1,2).contiguous())   #(B*h,L_q,L_k)
#         attention=torch.einsum('abcd,abed->abce',query_,key_)  #(B,h,L_q,L_k)
#         #
#         # if self.head_mode!=0:
#         #     attention = self.relu(attention)  # (B*h,L_q,L_k)
#
#         #Scale
#         attention=attention / (self.hidden_dim**0.5)   #(B,h,L_q,L_k)
#
#         #Key Mask
#         # if key_mask is None:
#         #     key_mask=key.abs().sum(dim=-1)   #(B,L_k)
#         if key_mask is not None:
#             key_mask=key_mask.eq(0) #(B,L_k)
#             key_mask=key_mask.unsqueeze(dim=1).repeat(1,self.head_num,1)   #(B,h,L_k)
#             key_mask=key_mask.unsqueeze(dim=2).expand(-1,-1,query.size(1),-1)    #(B,h,L_q,L_k)
#             attention=attention.masked_fill(key_mask,-np.inf)   #(B,h,L_q,L_k)
#
#         # if self.head_mode!=0:
#         #     attention=attention.masked_fill(attention<=0,-np.inf)    #(B*h,L_q,L_k)
#
#         # Causality = Future blinding
#         if self.causality:
#             seq_mask=torch.triu(torch.ones_like(attention[0,0,:,:],dtype=torch.uint8),diagonal=1) #(L_q,L_k)
#             # print(seq_mask)
#             # print(seq_mask.size())
#             # test_mat=torch.ones_like(attention[0,0,:,:])
#             # test_mat=test_mat.masked_fill(seq_mask,-np.inf)
#             seq_mask=seq_mask[None,None,:,:].expand(batch_size,self.head_num,-1,-1) #(B,h,L_q,L_k)
#             # print('seq_mask:',seq_mask.size())
#             # print('attention:',attention.size())
#             attention=attention.masked_fill(seq_mask,-np.inf)   #(B,h,L_q,L_k)
#             # print(attention.size())
#             # pass
#
#         #Softmax
#
#         attention = self.softmax(attention)  # (B,h,L_q,L_k)
#
#         if query_mask is not None:
#             query_mask=query_mask[:,None,:].repeat(1,self.head_num,1)  #(B,h,L_q)
#             query_mask=query_mask.unsqueeze(-1).expand(-1,-1,-1,value.size(1))   #(B,h,L_q,L_k)
#
#             # attention*=query_mask
#             attention=attention.mul(query_mask.float()) #(B,h,L_q,L_k)
#
#         #Dropouts
#         attention=self.dropout(attention)   #(B,h,L_q,L_k)
#         # if query.size(-1)>1:
#         # print('attention2:',attention[:,-3:,:])
#
#         # Weighted sum
#         output=torch.matmul(attention,value_)  #(B,h,L_q,D/h)
#         output=output.transpose(1,2).contiguous().view(batch_size,-1,self.head_num*self.head_dim)    #(B,L_q,D)
#         output=self.conv1d_o(output.transpose(1,2)).transpose(1,2)  #(B,L_q,D)
#         # # print(output)
#         # if self.residual:
#         #     # Residual connection
#         #     output=output.add(query)   #(B,L_q,D)
#         #
#         #     if self.head_mode!=0:
#         #         output = self.relu(output)  # (B,L_q,D)
#         #
#         # if self.norm:
#         #     # Normalize
#         #     output=self.layer_norm(output)  #(B,L_q,D)
#         #
#         # # print('query_mask:',query.abs().sum(dim=-1).sign())   #(B,L_q)
#         # # print('output_mask:',output.abs().sum(dim=-1).sign())   #(B,L_q)
#
#         return output
#
# class FeedForward(nn.Module):
#     def __init__(self,
#                  unit_num=512,
#                  hidden_dim=2048,
#                  drop_rate=0.,
#                  # residual=True,
#                  # norm=True
#                  ):
#         '''
#         :param unit_num: input and output dim
#         :param hidden_dim: hidden dim
#         :param drop_rate: A floating point number.
#         :param residual: Boolean. 是否加入残差
#         :param norm: Boolean. 是否归一化
#         '''
#         super().__init__()
#         # self.residual = residual
#         # self.norm = norm
#         self.conv1d1=nn.Conv1d(unit_num,hidden_dim,kernel_size=1,padding=0)
#         self.relu=nn.ReLU()
#         self.leaky_relu=nn.LeakyReLU()
#         self.conv1d2=nn.Conv1d(hidden_dim,unit_num,kernel_size=1)
#         self.dropout = nn.Dropout(drop_rate)
#         # self.layer_norm = nn.LayerNorm(unit_num)
#         # self.layer_norm = nn.LayerNorm(unit_num)
#
#     def forward(self, x,mask=None):
#         '''
#         Point-wise feed forward net
#         :param x: A 3d tensor with shape of [B, L, D].
#         :param mask: A 3d tensor with shape of [B, L].
#         :return: Boolean, whether to reuse the weights of a previous layer
#         by the same name.
#         '''
#         output = self.conv1d1(x.transpose(1, 2))  # (B,D,L)
#         output = self.relu(output)  # (B,D,L)
#         output = self.conv1d2(output).transpose(1, 2)  # (B,L,D)
#         # if mask is None:
#         #     mask = x.abs().sum(dim=-1).sign().float()  # (B,L)
#         if mask is not None:
#             mask = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))  # (B, L,D)
#             output=output.mul(mask.float()) # (B, L,D)
#
#         output=self.dropout(output) #(B,D,L)
#
#         # # output=self.relu(output)
#         # if self.residual:
#         #     # Residual connection
#         #     output=output.add(x)   #(B,L,D)
#         # if self.norm:
#         #     # Normalize
#         #     output=self.layer_norm(output)  #(B,L,D)
#         #     # output = self.leaky_relu(output)  # (B,L_q,D)
#         return output

if __name__=='__main__':
    a1=torch.rand(1,5,10)
    a2=torch.rand(1,5,10)
    attention=MultiHeadAttention(unit_num=10,
                             head_num=2,
                             drop_rate=0.)
    torch.autograd.set_detect_anomaly(True)
    a=attention(a1,a2)
    b=a.sum()
    b.backward()
    print(a)