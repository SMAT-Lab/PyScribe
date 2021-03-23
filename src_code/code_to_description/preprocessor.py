import json
import pickle
import os
import re
import numpy as np
from config import *
from collections import Counter
import sys
sys.path.append('../my_lib')
from my_ast import code2edges,code2ast_info
import nltk
import langid


def tokenize_english(text,vocabs=None,keep_punc=True,keep_stopword=True,lemmatize=True,lower=True,correct_dict=None):
    '''
    使用nltk分词
    :param text: 待分词的英文文本
    :param keep_punc: 是否保留标点符号
    :param keep_punc: 是否做词干提取，文本匹配时很有用
    :return: 分词后的词语列表
    '''
    # if lower:
    #     text=text.lower()
    # def replace_start(s,text):

    # text = text.replace("``", '"').replace("''", '"').replace('`', "'")
    # text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", r'<url>', text, re.S)

    # text=re_text(text)

    # code2description专用
    # text = re.sub(
    #     r'(--+)|(- (- )+)|(==+)|(= (= )+)|(##+)|(# (# )+)|(\*\*+)|(\* (\* )+)|(>>+)|(> (> )+)|(<<+)|(< (< )+)', '-- ',
    #     text, re.S)
    # text = re.sub(r'(--(--)+)|(-- (-- )+)', '-- ', text, re.S)

    words=nltk.word_tokenize(text)

    # code2description专用
    # words=' '.join(words).replace('< url >','<url>').split()
    text=' '.join(words)
    sp_marks = list(set(re.findall(r"<.*?>", text, re.S)))
    new_sp_marks=[sp_mark.replace(' ','') for sp_mark in sp_marks]
    sp_mark_tups=list(filter(lambda x: len(x[1])<6,zip(sp_marks,new_sp_marks)))
    for sp_mark,new_sp_mark in sp_mark_tups:
        text=text.replace(sp_mark,new_sp_mark)
    puncs = re.findall(r'\W+', text, re.S)
    for punc in sorted(list(set(puncs)),key=len):
        text=text.replace(punc,' '+punc+' ')
    # text = re.sub(r'(--+)|(- (- )+)|(==+)|(= (= )+)|(##+)|(# (# )+)|(\*\*+)|(\* (\* )+)|(>>+)|(> (> )+)|(<<+)|(< (< )+)', '-- ', text, re.S)
    # text = re.sub(r'(--(--)+)|(-- (-- )+)', '-- ', text, re.S)
    ########################################################################
    text=' '.join(text.split())
    text = text.replace("< url >", '<url>')
    text = text.replace('-', ' - ').replace('_', ' _ ').replace('.', ' . ')
    # 去掉引号
    text = text.replace("' s ", "^^^s ").replace("' t ", "^^^t ")
    text = re.sub(r'{}|{}'.format("'", '"'), '', text, re.S).strip()
    text = text.replace("^^^s ", "'s ").replace("^^^t ", "'t ")
    text=text.replace('( )','()')
    text=text.replace(':',',')
    ########################################################################
    words=text.split()  #单词纠错
    if correct_dict is not None and isinstance(correct_dict,dict):
        words=[correct_dict[word] if word in correct_dict.keys() else word for word in words]
        words=' '.join(words).split()
    if lemmatize:
        lemmatizer = nltk.stem.WordNetLemmatizer()  # 词干提取
        if vocabs is not None:
            words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word,pos='n'),pos='v'),pos='a')
                     if word not in vocabs else word for word in words]
        else:
            words = [lemmatizer.lemmatize(lemmatizer.lemmatize(lemmatizer.lemmatize(word,pos='n'),pos='v'),pos='a')
                     for word in words]

    if not keep_punc:
        stop_puncs_0 = ['|', '{}', '()', '[]', '&', '*',
                        '/', '//', '#', '\\', '~', '""', '‖', '§']
        stop_puncs_1 = ['、', '\'', '"', '.', ':', ',', '...', '{', '}', '(', ')', '[', ']',
                        ';', '?', '!', '-', '--']
        # stop_puncs_2 = ["``", "''",'`']
        stop_puncs = stop_puncs_0 + stop_puncs_1
        words=[word for word in words if word not in stop_puncs]
    if not keep_stopword:
        stop_words = nltk.corpus.stopwords.words('english')
        words=[word for word in words if word not in stop_words]
    if lower:
        if vocabs is not None:
            words=[word.lower() if word not in vocabs else word for word in words]
        else:
            words=[word.lower() for word in words]
    return words

def is_en(text):
    if langid.classify(text)[0]!='en':
        return False
    return True


def tokenize_raw_data(raw_data_path, token_data_path, correct_dict_path=None):
    logging.info(
        '########### Start tokenize data including tokenizing, tree processing, and number-identification transfering ##########')

    token_data_dir = os.path.dirname(token_data_path)
    if not os.path.exists(token_data_dir):
        os.makedirs(token_data_dir)
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    correct_dict = None
    if correct_dict_path is not None:
        with open(correct_dict_path) as f:
            correct_dict = json.load(f)

    for i, item in enumerate(raw_data):
        logging.info('------Process the %d-th item' % (i + 1))
        # token_texts=[]
        # for j,text in enumerate(item['texts']):
        words = tokenize_english(item['text'], vocabs=None, keep_punc=True, keep_stopword=True, lemmatize=True,
                                 lower=True, correct_dict=correct_dict)
        # if len(words)>70:
        #     print(item['id'])
        #     # break
        # assert len(words)<=70
        token_text = ' '.join(words)

        # token_texts.append(token_text)

        edge_starts, edge_ends, edge_depths, edge_lposes, edge_sposes = code2ast_info(item['code'], attribute='all')
        edges = ['(%s,%s)' % (edge_start, edge_end) for edge_start, edge_end in zip(edge_starts, edge_ends)]

        raw_data[i]['token_text'] = token_text
        raw_data[i]['nodes'] = ' '.join([edge_starts[0]] + edge_ends)
        raw_data[i]['node_poses'] = ' '.join(['(0,0,0)'] + ['({},{},{})'.format(x, y, z) for x, y, z in
                                                            zip(edge_depths, edge_lposes, edge_sposes)])
        # raw_data[i]['node_depths']=' '.join(['0']+[str(i+1) for i in edge_depths])
        # raw_data[i]['node_lposes']=' '.join(['0']+[str(i) for i in edge_lposes])
        # raw_data[i]['node_sposes']=' '.join(['0']+[str(i) for i in edge_sposes])
        # raw_data[i]['edge_starts'] = ' '.join(edge_starts)
        # raw_data[i]['edge_ends'] = ' '.join(edge_ends)
        raw_data[i]['edges'] = ' '.join(edges)
        raw_data[i]['edge_poses'] = ' '.join(['({},{},{})'.format(x, y, z) for x, y, z in
                                              zip(edge_depths, edge_lposes, edge_sposes)])
        # raw_data[i]['edge_depths'] = ' '.join([str(i) for i in edge_depths])
        # raw_data[i]['edge_lposes'] = ' '.join([str(i) for i in edge_lposes])
        # raw_data[i]['edge_sposes'] = ' '.join([str(i) for i in edge_sposes])
    with open(token_data_path, 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, indent=4, ensure_ascii=False)
    logging.info(
        '########### Finish tokenize data including tokenizing, tree processing, and number-identification transfering ##########')


def build_train_w2i2w(train_token_data_path,
                      node_w2i_path,
                      node_i2w_path,
                      edge_w2i_path,
                      edge_i2w_path,
                      ast_pos_w2i_path,
                      ast_pos_i2w_path,
                      # edge_pos_w2i_path,
                      # edge_pos_i2w_path,
                      word_w2i_path,
                      word_i2w_path,
                      in_min_token_count=3,
                      out_min_token_count=3
                      ):
    logging.info('########### Start building the dictionary of the training set ##########')
    dic_paths = [node_w2i_path,
                 node_i2w_path,
                 edge_w2i_path,
                 edge_i2w_path,
                 ast_pos_w2i_path,
                 ast_pos_i2w_path,
                 # edge_pos_w2i_path,
                 # edge_pos_i2w_path,
                 word_w2i_path,
                 word_i2w_path,
                 ]
    for dic_path in dic_paths:
        dic_dir = os.path.dirname(dic_path)
        if not os.path.exists(dic_dir):
            os.makedirs(dic_dir)

    with open(train_token_data_path, 'r', encoding='utf-8') as f:
        token_data = json.load(f)

    node_counter = Counter()
    edge_counter = Counter()
    ast_pos_counter = Counter()
    # edge_pos_counter=Counter()
    word_counter = Counter()
    for i, item in enumerate(token_data):
        logging.info('------Process the %d-th item' % (i + 1))
        node_counter += Counter(item['nodes'].split())
        edge_counter += Counter(item['edges'].split())
        ast_pos_counter += Counter(item['node_poses'].split()) + Counter(item['edge_poses'].split())
        # edge_pos_counter += Counter(item['edge_poses'].split())

        word_counter += Counter(item['token_text'].split())  # texts是一个列表
    # word_nodes=set(filter(lambda x: is_identified_word_token(x,prefix_operators), nodes))
    # word_nodes = set(filter(lambda x: node_counter[x] > min_count, word_nodes))
    # if '19961996' in nodes:
    #     print(item)
    # other_nodes=nodes-word_nodes
    # other_nodes=list(filter(lambda x: node_counter[x]>min_count, other_nodes))
    # nodes=list(word_nodes)+[OUT_BEGIN_TOKEN]+other_nodes
    general_vocabs = [PAD_TOKEN, UNK_TOKEN]

    nodes = list(filter(lambda x: node_counter[x] >= in_min_token_count, node_counter.keys()))
    nodes = general_vocabs + nodes

    edges = list(filter(lambda x: edge_counter[x] >= in_min_token_count, edge_counter.keys()))
    edges = general_vocabs + edges

    node_poses = list(filter(lambda x: ast_pos_counter[x] >= in_min_token_count, ast_pos_counter.keys()))
    node_poses = general_vocabs + node_poses

    # edge_poses = list(filter(lambda x: edge_pos_counter[x] >= in_min_token_count, edge_pos_counter.keys()))
    # edge_poses = general_vocabs + edge_poses

    words = list(filter(lambda x: word_counter[x] >= out_min_token_count, word_counter.keys()))
    words = general_vocabs + words + [OUT_END_TOKEN, OUT_BEGIN_TOKEN, ]

    node_indices = list(range(len(nodes)))
    edge_indices = list(range(len(edges)))
    ast_pos_indices = list(range(len(node_poses)))
    # edge_pos_indices = list(range(len(edge_poses)))
    words_indices = list(range(len(words)))

    node_w2i = dict(zip(nodes, node_indices))
    node_i2w = dict(zip(node_indices, nodes))
    edge_w2i = dict(zip(edges, edge_indices))
    edge_i2w = dict(zip(edge_indices, edges))
    ast_pos_w2i = dict(zip(node_poses, ast_pos_indices))
    ast_pos_i2w = dict(zip(ast_pos_indices, node_poses))
    # edge_pos_w2i = dict(zip(edge_poses, edge_pos_indices))
    # edge_pos_i2w = dict(zip(edge_pos_indices, edge_poses))

    word_w2i = dict(zip(words, words_indices))
    word_i2w = dict(zip(words_indices, words))

    dics = [node_w2i,
            node_i2w,
            edge_w2i,
            edge_i2w,
            ast_pos_w2i,
            ast_pos_i2w,
            # edge_pos_w2i,
            # edge_pos_i2w,
            word_w2i,
            word_i2w]
    for dic, dic_path in zip(dics, dic_paths):
        with open(dic_path, 'wb') as f:
            pickle.dump(dic, f)
        with open(dic_path + '.json', 'w') as f:
            json.dump(dic, f, indent=4, ensure_ascii=False)
    logging.info('########### Finish building the dictionary of the training set ##########')


def build_avail_data(token_data_path,
                     avail_data_path,
                     node_w2i_path,
                     edge_w2i_path,
                     ast_pos_w2i_path,
                     # edge_pos_w2i_path,
                     word_w2i_path):
    '''
    根据字典构建模型可用的数据集，数据集为一个列表，每个元素为一条数据，是由输入和输出两个元素组成的，
    输入元素为一个ndarray，每行分别为边起点、边终点、深度、全局位置、局部位置，
    输出元素为一个ndarray，为输出的后缀表达式
    :param token_data_path:
    :param avail_data_path:
    :param node_w2i_path:
    :param edge_depth_w2i_path:
    :param edge_lpos_w2i_path:
    :param edge_spos_w2i_path:
    :return:
    '''
    logging.info('########### Start building the train dataset available for the model ##########')
    avail_data_dir = os.path.dirname(avail_data_path)
    if not os.path.exists(avail_data_dir):
        os.makedirs(avail_data_dir)

    w2is = []
    for w2i_path in [node_w2i_path,
                     edge_w2i_path,
                     ast_pos_w2i_path,
                     # edge_pos_w2i_path,
                     word_w2i_path]:
        with open(w2i_path, 'rb') as f:
            w2is.append(pickle.load(f))
    node_w2i, edge_w2i, ast_pos_w2i, word_w2i = w2is
    unk_idx = w2is[0][UNK_TOKEN]
    pad_idx = w2is[0][PAD_TOKEN]
    with open(token_data_path, 'r') as f:
        token_data = json.load(f)

    avail_data = []
    in_node_counter = Counter()
    in_edge_counter = Counter()
    out_token_counter = Counter()
    w2is = [node_w2i,
            edge_w2i,
            ast_pos_w2i,
            ast_pos_w2i,
            word_w2i
            ]
    for i, item in enumerate(token_data):
        # logging.info('------Process the %d-th item' % (i+1))
        token_item = [item['nodes'],  # 0
                      item['edges'],  # 1
                      item['node_poses'],  # 2
                      item['edge_poses'],  # 3
                      item['token_text']
                      ]
        # w2is=[node_w2i,node_w2i,edge_depth_w2i, edge_lpos_w2i,edge_spos_w2i]
        avail_item = []
        for seq, w2i in zip(token_item, w2is):
            avail_item.append([w2i.get(token, unk_idx) for token in seq.split()])
        # tmp=np.array([avail_data[-1]]+avail_data[2:-1])
        # avail_data=[np.array(avail_data[:-1]),np.array([avail_data[-1]]+avail_data[2:-1])]
        avail_input1 = np.array([avail_item[0], avail_item[2]])  # node+ast_pos
        avail_input2 = np.array([avail_item[1], avail_item[3]])  # edge+edge_pos
        avail_input3 = np.array([avail_item[0], avail_item[2],
                                 avail_item[1] + [pad_idx], avail_item[3] + [pad_idx]])  # node+ast_pos+edge+edge_pos
        avail_output = np.array(avail_item[4])
        avail_data.append([avail_input1, avail_input2, avail_input3, avail_output])
        in_node_counter += Counter(avail_item[0])
        in_edge_counter += Counter(avail_item[1])
        out_token_counter += Counter(avail_item[4])
    # print(len(avail_data))
    # print(text_token_counter[UNK_TOKEN])
    logging.info('+++++++++ The ratios of unknown input node, input edge, and output token are:%f, %f, and %f' % (
        in_node_counter[unk_idx] / sum(in_node_counter.values()),
        in_edge_counter[unk_idx] / sum(in_edge_counter.values()),
        out_token_counter[unk_idx] / sum(out_token_counter.values()))
                 )
    with open(avail_data_path, 'wb') as f:
        pickle.dump(avail_data, f)
    logging.info('########### Finish building the train dataset available for the model ##########')

if __name__=='__main__':
    tokenize_raw_data(raw_data_path=train_raw_data_path,
                      token_data_path=train_token_data_path,
                      correct_dict_path=correct_dict_path)
    tokenize_raw_data(raw_data_path=valid_raw_data_path,
                      token_data_path=valid_token_data_path,
                      correct_dict_path=correct_dict_path)
    tokenize_raw_data(raw_data_path=test_raw_data_path,
                      token_data_path=test_token_data_path,
                      correct_dict_path=correct_dict_path)

    build_train_w2i2w(train_token_data_path=train_token_data_path,
                      node_w2i_path=node_w2i_path,
                      node_i2w_path=node_i2w_path,
                      edge_w2i_path=edge_w2i_path,
                      edge_i2w_path=edge_i2w_path,
                      ast_pos_w2i_path=ast_pos_w2i_path,
                      ast_pos_i2w_path=ast_pos_i2w_path,
                      # edge_pos_w2i_path=edge_pos_w2i_path,
                      # edge_pos_i2w_path=edge_pos_i2w_path,
                      word_w2i_path=word_w2i_path,
                      word_i2w_path=word_i2w_path,
                      in_min_token_count=in_min_token_count,
                      out_min_token_count=out_min_token_count)
    build_avail_data(token_data_path=train_token_data_path,
                     avail_data_path=train_avail_data_path,
                     node_w2i_path=node_w2i_path,
                     edge_w2i_path=edge_w2i_path,
                     ast_pos_w2i_path=ast_pos_w2i_path,
                     # edge_pos_w2i_path=edge_pos_w2i_path,
                     word_w2i_path=word_w2i_path)
    build_avail_data(token_data_path=valid_token_data_path,
                     avail_data_path=valid_avail_data_path,
                     node_w2i_path=node_w2i_path,
                     edge_w2i_path=edge_w2i_path,
                     ast_pos_w2i_path=ast_pos_w2i_path,
                     # edge_pos_w2i_path=edge_pos_w2i_path,
                     word_w2i_path=word_w2i_path)
    build_avail_data(token_data_path=test_token_data_path,
                     avail_data_path=test_avail_data_path,
                     node_w2i_path=node_w2i_path,
                     edge_w2i_path=edge_w2i_path,
                     ast_pos_w2i_path=ast_pos_w2i_path,
                     # edge_pos_w2i_path=edge_pos_w2i_path,
                     word_w2i_path=word_w2i_path)

