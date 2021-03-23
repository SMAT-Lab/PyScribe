#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# class LiTransEnc(nn.Module):
#     def __init__(self,
#                  query_dim=300,
#                  head_num=5,
#                  head_dim=None,
#                  layer_num=6,
#                  drop_rate=0.
#                  ):
#         '''
#
#         :param max_len: int，最大序列长度
#         :param query_dim: int，输入最后对维度（B,L,D)中的D
#         :param head_num: int，head的数量，要能被query_dim整除
#         :param head_mode: int，head模式（0，1，2）
#         :param layer_num: int，层数
#         :param drop_rate: float，drop rate
#         :param residual: Boolean. 是否加入残差
#         :param norm: Boolean. 是否归一化
#         '''
#         super().__init__()
#         self.query_dim = query_dim
#         self.head_num = head_num
#         self.head_dim=head_dim
#         # if head_dim is None:
#         self.head_dim=self.query_dim//self.head_num
#         self.layer_num = layer_num
#         self.drop_rate = drop_rate
#
#         self.token_forwards = nn.ModuleList(
#             [FeedForward(in_dim=query_dim,
#                          hid_dim=query_dim * 4,
#                          # head_num=head_num,
#                          drop_rate=drop_rate) for _ in range(layer_num)]
#         )
#         self.token_layer_norms1 = nn.ModuleList(
#             [nn.LayerNorm(query_dim, elementwise_affine=True) for _ in range(layer_num)]
#         )
#
#
#         self.token_attentions = nn.ModuleList(
#             [TAttention(query_dim=query_dim,
#                         head_num=head_num,
#                         head_dim=self.head_dim,
#                         drop_rate=drop_rate,
#                         # head_mode=1,
#                         causality=False) for _ in range(layer_num)]
#         )
#
#         self.global_layer_norms2 = nn.ModuleList(
#             [nn.LayerNorm(query_dim,elementwise_affine=True) for _ in range(layer_num)]
#         )
#
#         self.token_layer_norms2 = nn.ModuleList(
#             [nn.LayerNorm(query_dim,elementwise_affine=True) for _ in range(layer_num)]
#
#         )
#
#
#     def forward(self, x,mask):
#         '''
#
#         :param x: 3-D shaped tensor, (B,L,D)
#         :param mask: None or 2-D shaped tensor, (B,L)
#         :return: [batch, length, hidden] the output sequence
#                  [batch, hidden] the global relay node
#         '''
#         t=x
#         g_mask_2d = torch.ones(t.size(0), 1, device=t.device)  # (B,1)
#         t_len=t.size(1) #L
#         batch_max_len=mask.sum(1).max().int() #L-,实际最大长度（去0的）
#         t=t[:,:batch_max_len,:] #(B,L-,D)
#         t_mask_2d=mask[:,:batch_max_len] #(B,L-)
#         t_mask_3d = t_mask_2d.unsqueeze(-1).expand(-1, -1, self.query_dim)   #(B,L-,D)
#
#         g = t.max(dim=1)[0].unsqueeze(1)    #
#         for i in range(self.layer_num):
#
#             t_ = self.token_forwards[i](t, mask=t_mask_2d)  # (B,L-,D)
#             t = self.token_layer_norms1[i](t.add(t_))  # (B,L-,D)
#
#             t_ = self.token_attentions[i](t=t,
#                                           g=None,
#                                           t_mask=t_mask_2d,
#                                           g_mask=g_mask_2d,
#                                           )
#             t=self.token_layer_norms2[i](t.add(t_))    # (B,L-,D)
#             g = self.global_layer_norms2[i](g.add(t.max(dim=1)[0].unsqueeze(1)))
#         t = t.mul(t_mask_3d.float())  # (B,L-,D)
#         t=F.pad(t,(0,0,0,t_len-batch_max_len,0,0))  #padding，to (B,L,D)
#         return t,g.squeeze(1)   # (B,L,D),(B,D)

class LiTranEnc(nn.Module):
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
            query=self.enc_blocks[i](x=query,x_mask=query_mask)
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
        self.res_att=ResEncAtt(query_dim=query_dim,
                            head_num=head_num,
                            head_dim=head_dim,
                            drop_rate=drop_rate
                               )
        self.res_ff = ResFF(in_dim=query_dim,
                            hid_dim=ff_hid_dim,
                            # out_dim=query_dim,
                            drop_rate=drop_rate)
        # self.res_lstm = ResLSTM(in_dim=query_dim,
        #                         hid_dim=query_dim * 2,
        #                         drop_rate=drop_rate)
    def forward(self, x,x_mask):
        '''

        :param x: 3-D shaped tensor, (B,L,D)
        :return: [batch, length, hidden] the output sequence
                 [batch, hidden] the global relay node
        '''
        x=self.res_att(x=x, x_mask=x_mask)
        x=self.res_ff(x,mask=x_mask)
        # query=self.res_lstm(query,mask=query_mask)
        return x  # (B,L,D)
  
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

class ResEncAtt(nn.Module):
    def __init__(self,
                 query_dim=512,
                 head_num=8,
                 head_dim=None,
                 drop_rate=0.
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
        self.attention = EncAtt(query_dim=self.query_dim,
                                            head_num=self.head_num,
                                            head_dim=self.head_dim,
                                            drop_rate=drop_rate,)
        self.layer_norm = nn.LayerNorm(self.query_dim, elementwise_affine=True)

    def forward(self, x, x_mask):
        '''

        :param x: 3-D shaped tensor, (B,L,D)
        :return: [batch, length, hidden] the output sequence
                 [batch, hidden] the global relay node
        '''
        # x_len=x.size(1) #L
        x_ = self.attention(x=x, x_mask=x_mask)  # (B,L-,D)
        x = self.layer_norm(x_.add(x))
        # if query_mask is not None:
        #     query_mask = query_mask.unsqueeze(-1).repeat(1, 1, query.size(-1))  # (B, L,D)
        #     query=query.mul(query_mask.float()) # (B, L,D)
        return x  # (B,L,D)

class EncAtt(nn.Module):
    def __init__(self,
                 query_dim=512,
                 head_num=5,
                 head_dim=None,
                 drop_rate=0.,
                 ):
        '''
        init
        :param query_dim: A scalar. Attention size.
        :param head_num: An int. Number of heads.
        :param drop_rate: A floating point number.
        :param head_mode: An int. head模式，0表示标准transformer头，1表示共享Linear层transformer头，2表示共享卷积层transformer头
        :param residual: Boolean. 是否加入残差
        :param norm: Boolean. 是否归一化
        :param causality: Boolean. If true, units that reference the future are masked.
        '''
        super().__init__()
        self.query_dim=query_dim
        self.head_num=head_num
        self.head_dim=head_dim
        self.head_dim = query_dim // head_num
        self.hid_dim = self.head_num * self.head_dim
        self.conv1d_ins=nn.ModuleList([nn.Conv1d(query_dim, self.hid_dim, kernel_size=3,padding=1) for _ in range(3)])
        
        self.conv1d_gate=nn.Conv1d(query_dim,query_dim, kernel_size=1,padding=0)


        self.conv1d_out=nn.Conv1d(self.hid_dim,query_dim, kernel_size=3,padding=1)

        self.softmax=nn.Softmax(dim=-1)
        self.sigmoid=nn.Sigmoid()
        self.dropout=nn.Dropout(drop_rate)

    def forward(self,x,x_mask):
        '''
        Applies multihead attention
        :param x: A 3d tensor with shape of [B, L_q, D_q]
        :param g: A 3d tensor with shape of [B, 1, D_k]
        :param c: A 3d tensor with shape of [B, L_q, D_k]
        :param x_mask: A 3d tensor with shape of [B, L_q]
        :param g_mask: A 3d tensor with shape of [B, 1]
        :param c_mask: A 3d tensor with shape of [B, L_q]
        :return: A 3d tensor with shape of [B, L_q, D]
        '''
        batch_size=x.size(0)    #B
        query_len=x.size(1)     #L_q
        gate=self.sigmoid(self.conv1d_gate(x.transpose(1, 2)).transpose(1,2))
        query,key,value= [conv1d(x.transpose(1, 2)) for conv1d, x in
                                zip(self.conv1d_ins, (x, x, x))]  # (B,D,L),(B,D,L),(B,D,L)
        query_=query.view(batch_size,self.head_num,self.head_dim,-1,1).permute(0,1,3,4,2)  #(B,h,L,1,D/h)
        query_mask=x_mask.unsqueeze(-1).expand(-1,-1,self.query_dim).float()  #(B,L,1)

        key_t,value_t=[x.view(batch_size,self.head_num,self.head_dim,-1,1).permute(0,1,3,4,2)
                                     for x in (key,value)]  #for all (B,h,L,1,D/h)

        key_g, value_g = [x.max(dim=-1)[0][:, :, None, None].expand(-1, -1, query_len, -1).
                                view(batch_size, self.head_num, self.head_dim, query_len, -1).permute(0, 1, 3, 4, 2)
                            for x in (key, value)]  # for all (B,h,L,1,D/h)
        key_,value_=[torch.cat([x_t,x_g],dim=-2) for x_t,x_g
                     in ((key_t,key_g),(value_t,value_g))] #for all (B,h,L,3,D/h)


        # key_mask = g_mask.unsqueeze(1).expand(-1, query_len, -1).float()  # (B,L_q,1)
        # key_mask=torch.cat([query_mask,key_mask],dim=-1) #(B,L_q,3)


        # value_=key_*1.

        #Multiplication
        # attention=torch.matmul(query_,key_.transpose(-2,-1).contiguous())   #(B,h,L_q,1,1+L_k)
        attention=torch.einsum('abcde,abcfe->abcdf',query_,key_)  #(B,h,L_q,1,3)

        #Scale
        attention=attention / (self.hid_dim**0.5)   #(B,h,L_q,1,3)

        #Key Mask
        # if key_mask is not None:
        # key_mask_=key_mask[:,None,:,None,:].expand(-1,self.head_num,-1,1,-1)  #(B,h,L_q,1,3)
        # key_mask_ = key_mask_.eq(0)  #(B,h,L_q,1,3)
        # attention=attention.masked_fill(key_mask_,-np.inf)   #(B,h,L_q,1,3)

        # Causality = Future blinding
        # if self.causality:
        #     seq_mask=torch.triu(torch.ones_like(attention[0,:,:],dtype=torch.unit8),diagonal=1) #(L_q,L_k)
        #     seq_mask=seq_mask[None,None,:,:].expand(batch_size,self.head_num,-1,-1) #(B,h,L_q,L_k)
        #     attention=attention.masked_fill(seq_mask,-np.inf)   #(B,h,L_q,L_k)

        #Softmax
        attention = self.softmax(attention)  #(B,h,L_q,1,3)

        # if query_mask is not None:  #(B,L_q,1)
        # query_mask=query_mask[:,None,:,:,None].expand(-1,self.head_num,-1,-1,attention.size(-1))  #(B,h,L_q,1,3)
        # attention=attention.mul(query_mask.float()) #(B,h,L_q,1,3)

        #Dropouts
        attention=self.dropout(attention)   #(B,h,L_q,1,3)

        # Weighted sum  #(B,h,L_q,1,L_k+3) (B,h,L_q,L_v+2,D)
        # output=torch.matmul(attention,value_)  #(B,h,L_q,1,D/h)
        output=torch.einsum('abcde,abcef->abcdf',attention,value_)  #(B,h,L_q,1,D/h)
        output=output.squeeze(dim=-2)    #(B,h,L_q,D/h)

        # output=self.conv1d_h(output.view(batch_size,self.head_num,-1)).view(batch_size,self.head_num,-1,self.head_dim)  #(B,h,L_q,D/h)

        # Restore shape
        # output=torch.cat(output.split(split_size=batch_size,dim=0),dim=-1).contiguous()  #(B,L_q,D)
        output=output.transpose(1,2).contiguous().view(batch_size,-1,self.head_num*self.head_dim)   #(B,L_q,D)
        # output = output.mul(gate)
        output=self.conv1d_out(output.transpose(1,2)).transpose(1,2)  #(B,L_q,D)
        # output=self.dropout(output) ##(B,L_q,D)
        output=output.mul(gate).mul(query_mask)

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


class SAttention(nn.Module):
    def __init__(self,
                 unit_num=512,
                 head_num=8,
                 drop_rate=0.,
                 # head_mode=0,
                 residual=True,
                 norm=True,
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
        self.head_num=head_num
        self.head_dim=unit_num//head_num
        self.hid_dim = self.head_num * self.head_dim
        # self.head_mode=head_mode
        self.residual=residual
        self.norm=norm
        self.causality=causality
        self.conv1d_ins=nn.ModuleList([nn.Conv1d(unit_num, self.head_dim*head_num, kernel_size=3,padding=1) for _ in range(3)])
        self.conv1d_h=nn.Conv1d(self.head_num,self.head_num,kernel_size=1)
        self.conv1d_out=nn.Conv1d(self.head_dim*head_num,unit_num, kernel_size=1,padding=0)

        self.relu=nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.softmax=nn.Softmax(dim=-1)
        self.sigmoid=nn.Sigmoid()
        self.dropout=nn.Dropout(drop_rate)
        # self.layer_norm=nn.LayerNorm(unit_num)
        self.maxpool1d = nn.MaxPool1d(kernel_size=3, stride=1, padding=3//2)


    def forward(self,t,g,g_mask,t_mask):
        '''
        Applies multihead attention
        :param query: A 3d tensor with shape of [B, T_q, D_q]
        :param key: A 3d tensor with shape of [B, T_k, D_k]
        :param query_mask: A 3d tensor with shape of [B, T_q]
        :param key_mask: A 3d tensor with shape of [B, T_k]
        :return: A 3d tensor with shape of [B, T_q, D]
        '''
        # if value is None:
        #     value=key*1.
        batch_size = t.size(0)  # B
        query_len = 1  # L_q

        query, key, value = [conv1d(x.transpose(1, 2)) for conv1d, x in
                             zip(self.conv1d_ins, (t, t, t))]  # (B,D,L),(B,D,L),(B,D,L)
        query_=query.max(-1)[0].view(batch_size, self.head_num, self.head_dim, 1).transpose(2, 3) # (B,h,1,D/h)


        key_g=key.max(-1)[0].view(batch_size, self.head_num, self.head_dim, 1).transpose(2, 3) # (B,h,1,D/h)
        key_t=key.view(batch_size, self.head_num, self.head_dim, -1).transpose(2, 3)    #(B,h,L_k,D/h)
        key_c=self.maxpool1d(key).view(batch_size, self.head_num, self.head_dim, -1).transpose(2, 3)    #(B,h,L_k,D/h)

        key_=torch.cat([key_g,key_c],dim=-2)    #(B,h,1+L_k,D/h)

        value_g = value.max(-1)[0].view(batch_size, self.head_num, self.head_dim, 1).transpose(2, 3)  # (B,h,1,D/h)
        value_t = value.view(batch_size, self.head_num, self.head_dim, -1).transpose(2, 3)  # (B,h,L_k,D/h)
        value_c=self.maxpool1d(value).view(batch_size, self.head_num, self.head_dim, -1).transpose(2, 3)    #(B,h,L_k,D/h)
        value_ = torch.cat([value_g, value_c], dim=-2)  # (B,h,1+L_k,D/h)

        #Multiplication
        # attention=torch.matmul(query_,key_.transpose(-2,-1).contiguous())   #(B,h,1,1+L_k)
        attention=torch.einsum('abcd,abed->abce',query_,key_)  #(B,h,1,1+L_k)
        #
        # if self.head_mode!=0:
        #     attention = self.relu(attention)  # (B*h,L_q,L_k)

        #Scale
        attention=attention / (self.hid_dim**0.5)   #(B,h,1,1+L_k)

        #Key Mask
        # if key_mask is None:
        #     key_mask=key.abs().sum(dim=-1)   #(B,L_k)
        if t_mask is not None:
            key_mask=torch.cat([g_mask.float(),t_mask.float()],dim=-1)    #(B,1+L_k)
            key_mask_=key_mask.eq(0) #(B,1+L_k)
            key_mask_=key_mask_[:,None,None,:].expand(-1,self.head_num,query_len,-1)    #(B,h,1,1+L_k)
            attention=attention.masked_fill(key_mask_,-np.inf)   #(B,h,1,1+L_k)

        # if self.head_mode!=0:
        #     attention=attention.masked_fill(attention<=0,-np.inf)    #(B*h,L_q,L_k)

        # Causality = Future blinding
        if self.causality:
            seq_mask=torch.triu(torch.ones_like(attention[0,:,:],dtype=torch.unit8),diagonal=1) #(1,1+L_k)
            seq_mask=seq_mask[None,None,:,:].expand(batch_size,self.head_num,-1,-1) #(B,h,1,1+L_k)
            attention=attention.masked_fill(seq_mask,-np.inf)   #(B,h,1,1+L_k)

        #Softmax
        attention = self.softmax(attention)  # (B,h,1,1+L_k)

        if g_mask is not None:
            query_mask=g_mask[:,None,:,None].expand(-1,self.head_num,-1,key_.size(-2))  #(B,h,1,1+L_k)
            # attention*=query_mask
            attention=attention.mul(query_mask.float()) #(B,h,1,1+L_k)

        #Dropouts
        attention=self.dropout(attention)   #(B,h,1,1+L_k)

        # Weighted sum
        # output=torch.matmul(attention,value_)  #(B,h,1,D/h)
        output=torch.einsum('abcd,abdf->abcf',attention,value_)  #(B,h,1,D/h)

        output=self.conv1d_h(output.view(batch_size,self.head_num,-1)).view(batch_size,self.head_num,-1,self.head_dim)  #(B,h,1,D/h)
        output=output.transpose(1,2).contiguous().view(batch_size,-1,self.head_num*self.head_dim)    #(B,1,D)
        output=self.conv1d_out(output.transpose(1,2)).transpose(1,2)  #(B,1,D)

        return output