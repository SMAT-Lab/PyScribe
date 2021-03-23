import logging
import os
from torchtext.data.metrics import bleu_score

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#顶级数据目录
cur_data_dir='../../data/'

max_ast_size=220
max_text_size=70

raw_data_dir=os.path.join(cur_data_dir,'code_to_description_raw_data/')
train_raw_data_path=os.path.join(raw_data_dir,'train_data.json')
valid_raw_data_path=os.path.join(raw_data_dir,'valid_data.json')
test_raw_data_path=os.path.join(raw_data_dir,'test_data.json')

basic_info_dir=os.path.join(cur_data_dir,'basic_info/')
size_info_path=os.path.join(basic_info_dir,'ast_and_text_size.pkl')


token_data_dir=os.path.join(cur_data_dir,'token_data/')
train_token_data_path=os.path.join(token_data_dir,'train_data.json')
valid_token_data_path=os.path.join(token_data_dir,'valid_data.json')
test_token_data_path=os.path.join(token_data_dir,'test_data.json')

w2i2w_dir=os.path.join(cur_data_dir,'w2i2w/')
node_w2i_path=os.path.join(w2i2w_dir,'node_w2i.pkl')
node_i2w_path=os.path.join(w2i2w_dir,'node_i2w.pkl')
edge_w2i_path=os.path.join(w2i2w_dir,'edge_w2i.pkl')
edge_i2w_path=os.path.join(w2i2w_dir,'edge_i2w.pkl')
ast_pos_w2i_path=os.path.join(w2i2w_dir,'ast_pos_w2i.pkl')
ast_pos_i2w_path=os.path.join(w2i2w_dir,'ast_pos_i2w.pkl')
word_w2i_path=os.path.join(w2i2w_dir,'word_w2i.pkl')
word_i2w_path=os.path.join(w2i2w_dir,'word_i2w.pkl')

avail_data_dir=os.path.join(cur_data_dir,'avail_data/')
train_avail_data_path=os.path.join(avail_data_dir,'train_data.pkl')
valid_avail_data_path=os.path.join(avail_data_dir,'valid_data.pkl')
test_avail_data_path=os.path.join(avail_data_dir,'test_data.pkl')

correct_dict_path=os.path.join(basic_info_dir,'correction_dict.json')

OUT_BEGIN_TOKEN='</s>'
OUT_END_TOKEN='</e>'
PAD_TOKEN='<pad>'
UNK_TOKEN='<unk>'

in_min_token_count=3
out_min_token_count=3

train_data_num=45000
valid_data_num=1000
test_data_num=1000

emb_dir=os.path.join(cur_data_dir,'emb/')
model_dir=os.path.join(cur_data_dir,'model/')

eval_dir=os.path.join(cur_data_dir,'evaluation_result/')

########### Parameter Setting ##################
model_name = 'code2dsc_model'
model_id = None
enc_emb_dim = 512   # 512
dec_word_emb_dim = 512  # 512
enc_node_emb_path = None
enc_edge_emb_path = None
dec_word_emb_path = None
enc_node_emb_freeze = False
enc_edge_emb_freeze = False
dec_word_emb_freeze = False
enc_head_num = 8    #  8
dec_head_num = 8    # 8
enc_head_dim = None
dec_head_dim = None
enc_ff_hid_dim = 4 * enc_emb_dim    # 4*
dec_ff_hid_dim = 4 * dec_word_emb_dim   # 4*
enc_att_layer_num = 4   # 3
dec_att_layer_num = 4   # 3
enc_drop_rate = 0.2
dec_drop_rate = 0.2
batch_size =160
big_epochs = 20
regular_rate = 1e-5
lr_base = 5e-4
lr_decay = 0.95
min_lr_rate = 0.05
warm_big_epochs = 3
fit_log_verbose=1

train=True #if the model has been trained, you can set it to "False" to predict directly, otherwise set it to "True"

import sys
sys.path.append('../my_lib')
from my_evaluation import *
train_metrics = [bleu_score]
valid_metric = bleu_score
test_metrics = None #[bleu_score]


import random
import torch
import numpy as np
import os
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)    # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)    # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True

seed_torch(0)
