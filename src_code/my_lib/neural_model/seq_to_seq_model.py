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
import torch.nn.functional as F
import numpy as np
import os
import logging
import pickle
from datetime import datetime
import codecs
import copy
from torchtext.data.metrics import bleu_score
# from nltk.translate import meteor_score
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
import pandas as pd
import math


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class TransSeq2Seq(BaseModel):
    def _do_validation(self, valid_ins=None, valid_outs=None, last=False,
                       increase_better=False):
        '''
        根据验证集选择最好模型
        :param criterion: func，计算loss的评价函数
        # :param Dataset: class，torch的定制Dataset类
        :param valid_ins: lists, 验证集特征
        :param valid_outs: list, 验证集输出
        :param last: boolen, 是否为最后一次，用在训练轮次结束后，用于选出在验证集表现最好的模型做为最终模型
        :param increase_better: boolen, 根据验证集选择更好模型时，指标是越大越好还是越小越好
        :param seq_mode:序列模式，'POS'或者None为普通序列分类问题（如词性标注），'NER'为序列标注问题（可能多个span label合并），
                'WHOLE'为整个序列是否全对的分类问题,'BLEU'为文本翻译评价
        :return:
        '''
        # self.net.eval()
        if not last and valid_ins is not None and valid_outs is not None:  # 如果有验证集
            if 'best_net' not in self.__dict__:
                # self.best_net = copy.deepcopy(self.net)
                self.best_net='Sure thing'
                self.valid_loss_val = 1000
                if increase_better:
                    self.valid_evals = [-1000] * (len(self.train_metrics) + 1)
                else:
                    self.valid_evals = [1000] * (len(self.train_metrics) + 1)
            pred_out_np,_=self.predict(valid_ins)   #(B,L),(B,D,L)

            #计算指标
            log_info = 'Comparison of previous and current '

            metrics = copy.deepcopy(self.train_metrics) #如果不深拷贝，self.train_metrics下面会跟着变化
            if self.valid_metric is not None:  # 如果有valid metric
                metrics.append(self.valid_metric)
            if isinstance(valid_outs[0][0],list):
                true_out_list = [[[self.out_i2w[idx] for idx in (true_out_item[:true_out_item.tolist().index(0)]
                                   if 0 in true_out_item else true_out_item)] for true_out_item in true_out] for true_out in valid_outs]  # (BL-,)
            else:
                true_out_list=[[[self.out_i2w[idx] for idx in (true_out[:true_out.tolist().index(0)]
                                   if 0 in true_out else true_out)]] for true_out in valid_outs]
            pred_out_list = [[self.out_i2w[idx] for idx in (pred_out[:pred_out.tolist().index(0)]
                                if 0 in pred_out else pred_out)] for pred_out in pred_out_np]

            valid_evals = [metric(pred_out_list, true_out_list) for metric in metrics]

            for i, metric in enumerate(metrics):
                log_info += '{}: ({},{}) # '.format(metric.__name__, self.valid_evals[i], valid_evals[i])

            logging.info(log_info)
            is_better = False
            if increase_better and valid_evals[-1] >= self.valid_evals[-1]:\
                is_better = True
            elif not increase_better and valid_evals[-1] <= self.valid_evals[-1]:\
                is_better = True

            if is_better:  # 如果表现更好
                self.valid_evals = valid_evals
                torch.save(self.net.state_dict(),os.path.join(self.model_dir,self.model_name+'_best_net.net'))
        elif last:
            self.net.load_state_dict(torch.load(os.path.join(self.model_dir,self.model_name+'_best_net.net')))
            self.net.train()

    def eval_seq(self,
                 test_ins,
                 test_outs,
                 test_metrics=[bleu_score],
                 ):
        '''
        序列标注的模型评价
        :param test_ins: 测试样本特征集二维数组
        :param test_outs: 类别标签一维数组
        :param unique_outs: 不同种类的标记列表
        :param test_metrics: 评价方法列表
        :param focus_labels: 关注类别列表
        :return:
        '''
        # logging.info('Evaluate %s' % self.model_name)
        pred_outs,_ = self.predict(test_ins)  # 预测出的标记二维数组
        if isinstance(test_outs[0][0],list):
            true_out_list = [[[self.out_i2w[idx] for idx in (test_out_item[:test_out_item.tolist().index(0)]
                                                             if 0 in test_out_item else test_out_item)]
                              for test_out_item in test_out] for test_out in test_outs]  # (BL-,)
        else:
            true_out_list = [[[self.out_i2w[idx] for idx in (true_out[:true_out.tolist().index(0)]
                                                             if 0 in true_out else true_out)]]
                             for true_out in test_outs]


        pred_out_list = [[self.out_i2w[idx] for idx in (pred_out[:pred_out.tolist().index(0)]
                                                        if 0 in pred_out else pred_out)] for pred_out in pred_outs]
        for i, pred_out in enumerate(pred_out_list):
            if len(pred_out) == 0:
                pred_out_list[i] = ['.']
            assert len(pred_out_list[i])>0
        eval_dic = dict()
        logging.info('Evaluate %s' % self.model_name)
        if test_metrics is None:
            for weight,metric_name in zip([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.],[0.25]*4],
                                            ['BLEU-1','BLEU-2','BLEU-3','BLEU-4','av-BLEU']):
                eval_res=bleu_score(pred_out_list,true_out_list,max_n=4, weights=weight)
                eval_dic[metric_name] = {'OVERALL':eval_res*100}
            for metric,metric_name in zip([Meteor(),Rouge(),Cider()],['METEOR','ROUGE','CIDER']):
                # print('metric_name')
                eval_res,_=metric.compute_score(pred_out_list, true_out_list)
                eval_dic[metric_name] = {'OVERALL': eval_res*100}
            # eval_res, eval_results = Bleu(n=4).compute_score(pred_out_list, true_out_list)
            # eval_dic['blue-{}'.format(i+1)] = {'OVERALL': eval_res}
            # print(eval_res, eval_results)
            # for metric,metric_name in zip([Bleu(n=4),Cider(),Meteor(),Rouge()],[''])
        else:
            for metric in test_metrics:

                eval_res = metric(pred_out_list,true_out_list)
                eval_dic[metric.__name__] = dict()
                if isinstance(eval_res, float) or isinstance(eval_res, int):
                    eval_dic[metric.__name__]['OVERALL'] = eval_res
                elif isinstance(eval_res, pd.Series):  # 如果评价结果是一个Series
                    eval_dic[metric.__name__] = dict(eval_res)
        eval_df = pd.DataFrame(eval_dic)
        return eval_df

    def eval_seq_by_lens(self,
                         test_ins,
                         test_outs,
                         eval_dir,
                         decrease_in=0,
                         in_sec_num=10,
                         in_min_len=0,
                         out_sec_num=10,
                         out_min_len=0,
                         ):
        logging.info('Predict the outputs of %s' % self.model_name)
        pred_outs, _ = self.predict(test_ins)  # 预测出的标记二维数组
        if not isinstance(test_outs[0][0],list):
            test_outs=[[test_out] for test_out in test_outs]
        # if isinstance(test_outs[0][0], list):
        true_out_list = [[[self.out_i2w[idx] for idx in (test_out_item[:test_out_item.tolist().index(0)]
                                                         if 0 in test_out_item else test_out_item)]
                          for test_out_item in test_out] for test_out in test_outs]  # (BL-,)
        # else:
        #     true_out_list = [[[self.out_i2w[idx] for idx in (true_out[:true_out.tolist().index(0)]
        #                                                      if 0 in true_out else true_out)]]
        #                      for true_out in test_outs]
        pred_out_list = [[self.out_i2w[idx] for idx in (pred_out[:pred_out.tolist().index(0)]
                                                        if 0 in pred_out else pred_out)] for pred_out in pred_outs]
        # in_max_len = max([len(seqs[0]) for seqs in train_ins])  # 最大输入长度
        # out_max_len = max([len(seq) for seq in train_outs])  # 最大输出长度
        data_num = len(true_out_list)  # 数据数量

        logging.info('--- Count the input lengths')

        in_lens = [len(seqs[0]) - decrease_in for seqs in test_ins]
        max_in_len=max(in_lens)
        interval = math.ceil(1.0*max_in_len/in_sec_num)
        in_lens=['({},{}]'.format(((in_len-1)//interval)*interval,((in_len-1)//interval+1)*interval) for in_len in in_lens]
        # in_lens=[((in_len-1)//interval+1)*interval for in_len in in_lens]
        in_len2out = dict()
        for i, in_len in enumerate(in_lens):
            if in_len not in in_len2out:
                in_len2out[in_len] = [[pred_out_list[i]], [true_out_list[i]], 1]
            else:
                in_len2out[in_len][0].append(pred_out_list[i])
                in_len2out[in_len][1].append(true_out_list[i])
                in_len2out[in_len][2] += 1

        logging.info('--- Count the output lengths')

        out_lenss = [list(set([len(seq) for seq in test_out_seqs])) for test_out_seqs in test_outs]
        max_out_len = max([max(out_lens) for out_lens in out_lenss])
        interval = math.ceil(1.0*max_out_len/out_sec_num)
        out_lenss = [['({},{}]'.format(((out_len-1) // interval) * interval, ((out_len-1) // interval + 1) * interval) for out_len in
                   out_lens] for out_lens in out_lenss]
        # out_lenss = [[((out_len-1) // interval + 1) * interval for out_len in
        #            out_lens] for out_lens in out_lenss]
        out_len2out = dict()
        for i, out_lens in enumerate(out_lenss):
            for out_len in out_lens:
                if out_len not in out_len2out:
                    out_len2out[out_len] = [[pred_out_list[i]], [true_out_list[i]], 1]
                else:
                    out_len2out[out_len][0].append(pred_out_list[i])
                    out_len2out[out_len][1].append(true_out_list[i])
                    out_len2out[out_len][2] += 1

        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)

        # res_paths=[os.path.join(eval_dir,'bleu_by_ins.xlsx'),
        #            os.path.join(eval_dir,'meteor_by_ins.xlsx'),
        #            os.path.join(eval_dir,'bleu_by_outs.xlsx'),
        #            os.path.join(eval_dir,'meteor_by_outs.xlsx')]
        # metrics=[]
        eval_vs_in_len_path = os.path.join(eval_dir, 'eval_vs_in_len.xlsx')
        logging.info('--- Evaluate the av-BLEU of {} and save the evaluation_result by input length into {}.'.
                     format(self.model_name, eval_vs_in_len_path))
        if not os.path.exists(eval_vs_in_len_path):
            eval_vs_in_len_df = pd.DataFrame(columns=sorted(in_len2out.keys(),key=lambda x:int(x.split(',')[0][1:])))
        else:
            eval_vs_in_len_df = pd.read_excel(eval_vs_in_len_path, index_col=0, header=0)

        for in_len in in_len2out.keys():
            eval_vs_in_len_df.loc['Proportion', in_len] = 1.0 * in_len2out[in_len][2] / data_num * 100
            # eval_vs_in_len_df.loc['av-BLEU', in_len] = bleu_score(in_len2out[in_len][0], in_len2out[in_len][1]) * 100
            for weight,metric_name in zip([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.],[0.25]*4],
                                            ['BLEU-1','BLEU-2','BLEU-3','BLEU-4','av-BLEU']):
                eval_res=bleu_score(in_len2out[in_len][0],in_len2out[in_len][1],max_n=4, weights=weight)
                eval_vs_in_len_df.loc[metric_name, in_len] = eval_res * 100
            for metric, metric_name in zip([Meteor(), Rouge(), Cider()], ['METEOR', 'ROUGE', 'CIDER']):
                # print(len(in_len2out[in_len][0]))
                # print(len(in_len2out[in_len][1]))
                # print(metric_name)
                eval_res, _ = metric.compute_score(in_len2out[in_len][0], in_len2out[in_len][1])
                eval_vs_in_len_df.loc[metric_name, in_len] = eval_res * 100
        eval_vs_in_len_df.to_excel(eval_vs_in_len_path)

        eval_vs_out_len_path = os.path.join(eval_dir, 'eval_vs_out_len.xlsx')
        logging.info('--- Evaluate the av-BLEU of {} and save the evaluation_result by input length into {}.'.
                     format(self.model_name, eval_vs_out_len_path))
        if not os.path.exists(eval_vs_out_len_path):
            eval_vs_out_len_df = pd.DataFrame(columns=sorted(out_len2out.keys(),key=lambda x:int(x.split(',')[0][1:])))
        else:
            eval_vs_out_len_df = pd.read_excel(eval_vs_out_len_path, index_col=0, header=0)

        for out_len in out_len2out.keys():
            eval_vs_out_len_df.loc['Proportion', out_len] = 1.0 * out_len2out[out_len][2] / data_num * 100
            # eval_vs_out_len_df.loc['av-BLEU', out_len] = bleu_score(out_len2out[out_len][0],out_len2out[out_len][1]) * 100
            for weight,metric_name in zip([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.],[0.25]*4],
                                            ['BLEU-1','BLEU-2','BLEU-3','BLEU-4','av-BLEU']):
                eval_res=bleu_score(out_len2out[out_len][0],out_len2out[out_len][1],max_n=4, weights=weight)
                eval_vs_out_len_df.loc[metric_name, out_len] = eval_res * 100
            for metric, metric_name in zip([Meteor(), Rouge(), Cider()], ['METEOR', 'ROUGE', 'CIDER']):
                # print('metric_name')
                eval_res, _ = metric.compute_score(out_len2out[out_len][0], out_len2out[out_len][1])
                eval_vs_out_len_df.loc[metric_name, out_len] = eval_res * 100
        eval_vs_out_len_df.to_excel(eval_vs_out_len_path)

        return eval_vs_in_len_df, eval_vs_out_len_df

    # def eval_seq_by_lens(self,
    #                      test_ins,
    #                      test_outs,
    #                      eval_dir,
    #                      decrease_in=0,
    #                      ):
    #     logging.info('Predict the outputs of %s' % self.model_name)
    #     pred_outs, _ = self.predict(test_ins)  # 预测出的标记二维数组
    #     true_out_list = [[[self.out_i2w[idx] for idx in (test_out_item[:test_out_item.tolist().index(0)]
    #                                                      if 0 in test_out_item else test_out_item)] for test_out_item in
    #                       test_out]
    #                      for test_out in test_outs]  # (BL-,)
    #     pred_out_list = [[self.out_i2w[idx] for idx in (pred_out[:pred_out.tolist().index(0)]
    #                                                     if 0 in pred_out else pred_out)] for pred_out in pred_outs]
    #     # in_max_len = max([len(seqs[0]) for seqs in train_ins])  # 最大输入长度
    #     # out_max_len = max([len(seq) for seq in train_outs])  # 最大输出长度
    #     data_num = len(true_out_list)  # 数据数量
    #
    #     logging.info('--- Count the input lengths')
    #     in_lens = [len(seqs[0]) - decrease_in for seqs in test_ins]
    #     in_len2out = dict()
    #     for i, in_len in enumerate(in_lens):
    #         if in_len not in in_len2out:
    #             in_len2out[in_len] = [[pred_out_list[i]], [true_out_list[i]], 1]
    #         else:
    #             in_len2out[in_len][0].append(pred_out_list[i])
    #             in_len2out[in_len][1].append(true_out_list[i])
    #             in_len2out[in_len][2] += 1
    #
    #     logging.info('--- Count the output lengths')
    #     out_lenss = [list(set([len(seq) for seq in test_out_seqs])) for test_out_seqs in test_outs]
    #     out_len2out = dict()
    #     for i, out_lens in enumerate(out_lenss):
    #         for out_len in out_lens:
    #             if out_len not in out_len2out:
    #                 out_len2out[out_len] = [[pred_out_list[i]], [true_out_list[i]], 1]
    #             else:
    #                 out_len2out[out_len][0].append(pred_out_list[i])
    #                 out_len2out[out_len][1].append(true_out_list[i])
    #                 out_len2out[out_len][2] += 1
    #
    #     if not os.path.exists(eval_dir):
    #         os.makedirs(eval_dir)
    #
    #     # res_paths=[os.path.join(eval_dir,'bleu_by_ins.xlsx'),
    #     #            os.path.join(eval_dir,'meteor_by_ins.xlsx'),
    #     #            os.path.join(eval_dir,'bleu_by_outs.xlsx'),
    #     #            os.path.join(eval_dir,'meteor_by_outs.xlsx')]
    #     # metrics=[]
    #     eval_vs_in_len_path=os.path.join(eval_dir, 'eval_vs_in_len.xlsx')
    #     logging.info('--- Evaluate the av-BLEU of {} and save the evaluation_result by input length into {}.'.
    #                  format(self.model_name, eval_vs_in_len_path))
    #     if not os.path.exists(eval_vs_in_len_path):
    #         eval_vs_in_len_df = pd.DataFrame(columns=sorted(in_len2out.keys()))
    #     else:
    #         eval_vs_in_len_df = pd.read_excel(eval_vs_in_len_path, index_col=0, header=0)
    #
    #     for in_len in sorted(in_len2out.keys()):
    #         eval_vs_in_len_df.loc['Proportion', in_len] = 1.0 * in_len2out[in_len][2] / data_num * 100
    #         eval_vs_in_len_df.loc['av-BLEU', in_len] = bleu_score(in_len2out[in_len][0],in_len2out[in_len][1]) * 100
    #         for metric, metric_name in zip([Meteor(), Rouge(), Cider()], ['METEOR', 'ROUGE', 'CIDER']):
    #             # print(len(in_len2out[in_len][0]))
    #             # print(len(in_len2out[in_len][1]))
    #             # print(metric_name)
    #             eval_res,_ = metric.compute_score(in_len2out[in_len][0],in_len2out[in_len][1])
    #             eval_vs_in_len_df.loc[metric_name, in_len]=eval_res*100
    #     eval_vs_in_len_df.to_excel(eval_vs_in_len_path)
    #
    #     eval_vs_out_len_path = os.path.join(eval_dir, 'eval_vs_out_len.xlsx')
    #     logging.info('--- Evaluate the av-BLEU of {} and save the evaluation_result by input length into {}.'.
    #                  format(self.model_name, eval_vs_out_len_path))
    #     if not os.path.exists(eval_vs_out_len_path):
    #         eval_vs_out_len_df = pd.DataFrame(columns=sorted(out_len2out.keys()))
    #     else:
    #         eval_vs_out_len_df = pd.read_excel(eval_vs_out_len_path, index_col=0, header=0)
    #
    #     for out_len in sorted(out_len2out.keys()):
    #         eval_vs_out_len_df.loc['Proportion', out_len] = 1.0 * out_len2out[out_len][2] / data_num * 100
    #         eval_vs_out_len_df.loc['av-BLEU', out_len] = bleu_score(out_len2out[out_len][0], out_len2out[out_len][1]) * 100
    #         for metric, metric_name in zip([Meteor(), Rouge(), Cider()], ['METEOR', 'ROUGE', 'CIDER']):
    #             # print('metric_name')
    #             eval_res,_ = metric.compute_score(out_len2out[out_len][0], out_len2out[out_len][1])
    #             eval_vs_out_len_df.loc[metric_name, out_len]=eval_res*100
    #     eval_vs_out_len_df.to_excel(eval_vs_out_len_path)
    #
    #     return eval_vs_in_len_df, eval_vs_out_len_df

    # def eval_seq_by_lens(self,
    #              test_ins,
    #              test_outs,
    #              eval_dir,
    #                  decrease_in=0,
    #              ):
    #     logging.info('Predict the outputs of %s' % self.model_name)
    #     pred_outs, _ = self.predict(test_ins)  # 预测出的标记二维数组
    #     true_out_list = [[[self.out_i2w[idx] for idx in (test_out_item[:test_out_item.tolist().index(0)]
    #                                                      if 0 in test_out_item else test_out_item)] for test_out_item in
    #                       test_out]
    #                      for test_out in test_outs]  # (BL-,)
    #     pred_out_list = [[self.out_i2w[idx] for idx in (pred_out[:pred_out.tolist().index(0)]
    #                                                     if 0 in pred_out else pred_out)] for pred_out in pred_outs]
    #     # in_max_len = max([len(seqs[0]) for seqs in train_ins])  # 最大输入长度
    #     # out_max_len = max([len(seq) for seq in train_outs])  # 最大输出长度
    #     data_num=len(true_out_list) #数据数量
    #
    #     logging.info('--- Count the input lengths')
    #     in_lens=[len(seqs[0])-decrease_in for seqs in test_ins]
    #     in_len2out=dict()
    #     for i,in_len in enumerate(in_lens):
    #         if in_len not in in_len2out:
    #             in_len2out[in_len]=[[pred_out_list[i]],[true_out_list[i]],1]
    #         else:
    #             in_len2out[in_len][0].append(pred_out_list[i])
    #             in_len2out[in_len][1].append(true_out_list[i])
    #             in_len2out[in_len][2]+=1
    #
    #     logging.info('--- Count the output lengths')
    #     out_lenss = [list(set([len(seq) for seq in test_out_seqs])) for test_out_seqs in test_outs]
    #     out_len2out=dict()
    #     for i,out_lens in enumerate(out_lenss):
    #         for out_len in out_lens:
    #             if out_len not in out_len2out:
    #                 out_len2out[out_len]=[[pred_out_list[i]],[true_out_list[i]],1]
    #             else:
    #                 out_len2out[out_len][0].append(pred_out_list[i])
    #                 out_len2out[out_len][1].append(true_out_list[i])
    #                 in_len2out[out_len][2] += 1
    #
    #     if not os.path.exists(eval_dir):
    #         os.makedirs(eval_dir)
    #
    #     # res_paths=[os.path.join(eval_dir,'bleu_by_ins.xlsx'),
    #     #            os.path.join(eval_dir,'meteor_by_ins.xlsx'),
    #     #            os.path.join(eval_dir,'bleu_by_outs.xlsx'),
    #     #            os.path.join(eval_dir,'meteor_by_outs.xlsx')]
    #     # metrics=[]
    #     bleu_by_ins_path=os.path.join(eval_dir,'bleu_by_ins.xlsx')
    #     logging.info('--- Evaluate the av-BLEU of {} and save the evaluation_result by input length into {}.'.
    #                  format(self.model_name, bleu_by_ins_path))
    #     if not os.path.exists(bleu_by_ins_path):
    #         bleu_by_ins_df=pd.DataFrame(columns=sorted(in_len2out.keys()))
    #     else:
    #         bleu_by_ins_df=pd.read_excel(bleu_by_ins_path,index_col=0,header=0)
    #     for in_len in sorted(in_len2out.keys()):
    #         bleu_by_ins_df.loc['proportion', in_len]=1.0*in_len2out[in_len][2]/data_num*100
    #         bleu_by_ins_df.loc[self.model_name,in_len]=bleu_score(in_len2out[in_len][0],in_len2out[in_len][1])*100
    #     bleu_by_ins_df.to_excel(bleu_by_ins_path)
    #
    #     meteor_by_ins_path = os.path.join(eval_dir, 'meteor_by_ins.xlsx')
    #     logging.info('--- Evaluate the METEOR of {} and save the evaluation_result by input length into {}.'.
    #                  format(self.model_name, meteor_by_ins_path))
    #     if not os.path.exists(meteor_by_ins_path):
    #         meteor_by_ins_df = pd.DataFrame(columns=sorted(in_len2out.keys()))
    #     else:
    #         meteor_by_ins_df = pd.read_excel(meteor_by_ins_path,index_col=0,header=0)
    #     for in_len in sorted(in_len2out.keys()):
    #         meteor_by_ins_df.loc['proportion', in_len] = 1.0 * in_len2out[in_len][2] / data_num * 100
    #         meteor_by_ins_df.loc[self.model_name, in_len],_ = Meteor().compute_score(in_len2out[in_len][0], in_len2out[in_len][1])*100
    #     meteor_by_ins_df.to_excel(meteor_by_ins_path)
    #
    #     bleu_by_outs_path = os.path.join(eval_dir, 'bleu_by_outs.xlsx')
    #     logging.info('--- Evaluate the av-BLEU of {} and save the evaluation_result by output length into {}.'.
    #                  format(self.model_name, bleu_by_outs_path))
    #     if not os.path.exists(bleu_by_outs_path):
    #         bleu_by_outs_df = pd.DataFrame(columns=sorted(out_len2out.keys()))
    #     else:
    #         bleu_by_outs_df = pd.read_excel(bleu_by_outs_path,index_col=0,header=0)
    #     for out_len in sorted(out_len2out.keys()):
    #         bleu_by_outs_df.loc['proportion', out_len] = 1.0 * out_len2out[out_len][2] / data_num * 100
    #         bleu_by_outs_df.loc[self.model_name, out_len] = bleu_score(out_len2out[out_len][0], out_len2out[out_len][1])*100
    #     bleu_by_outs_df.to_excel(bleu_by_outs_path)
    #
    #     meteor_by_outs_path = os.path.join(eval_dir, 'meteor_by_outs.xlsx')
    #     logging.info('--- Evaluate the METEOR of {} and save the evaluation_result by output length into {}.'.
    #                  format(self.model_name, meteor_by_outs_path))
    #     if not os.path.exists(meteor_by_outs_path):
    #         meteor_by_outs_df = pd.DataFrame(columns=sorted(out_len2out.keys()))
    #     else:
    #         meteor_by_outs_df = pd.read_excel(meteor_by_outs_path,index_col=0,header=0)
    #     for out_len in sorted(out_len2out.keys()):
    #         meteor_by_outs_df.loc['proportion', out_len] = 1.0 * out_len2out[out_len][2] / data_num * 100
    #         meteor_by_outs_df.loc[self.model_name, out_len],_ = Meteor().compute_score(out_len2out[out_len][0],
    #                                                                                out_len2out[out_len][1])*100
    #     meteor_by_outs_df.to_excel(meteor_by_outs_path)
    #     return bleu_by_ins_path,meteor_by_ins_path,bleu_by_outs_path,meteor_by_outs_path


    def predict(self,ins,beam_width=1):
        '''
        预测样本的类别标签的接口
        :param ins: 样本特征集二维数组
        :return: 预测出的类别标签一维数组,或值
        '''
        if beam_width==1:
            pred_out_np,pred_out_prob_np=super().predict(ins)
            # print(pred_out_np[:3,:])
            for i,pred_outs in enumerate(pred_out_np):
                for j,out_idx in enumerate(pred_outs):    #prefix
                    if out_idx==self.out_end_idx:
                        break

                pred_out_np[i,j:]=0
        else:
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
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择GPU优先
            # beam_width = 1
            with torch.no_grad():
                # pred_out_probs = []
                for batch_features, batch_out_inputs in data_loader:
                    batch_features = batch_features.to(device)  # (B,L_x)
                    batch_out_inputs = batch_out_inputs.to(device)  # (B,L_y)
                    # print(batch_features.size())
                    # batch_encoder=self.net.module.encoding(batch_features) #(B,L_x,D_x)
                    batch_encoder = self.net.module.encoder(batch_features)  # (B,L_x,D_x)
                    for i in range(batch_features.size(0)):
                        i_encoder = batch_encoder[i, :, :].unsqueeze(0).expand(beam_width,-1,-1)  # (BW,L_x,D_x)
                        i_out_inputs = batch_out_inputs[i, :].unsqueeze(0).expand(beam_width,-1)  # (BW,L_y)
                        i_pred_out_probs = torch.zeros((beam_width, self.out_dim, self.out_max_len)).to(device)  # (BW,D_y,L_y)
                        i_acc_log_probs = torch.zeros((beam_width)).to(device)  # (BW,)
                        for j in range(self.out_max_len + 1):
                            j_pred_out_probs = self.net.module.decoder(i_out_inputs,i_encoder)  # (BW,D_y,L_y)
                            bw_out_probs=[]
                            for bw in range(beam_width):
                                bw_out_probs=1


                        # beam_width = 10
                        com_i_out_probs = []
                        com_mean_i_log_probs = []
                        tmp_i_encoder = batch_encoder[i, :, :].unsqueeze(0)  # (1,L_x,D_x)
                        tmp_i_out_inputs = batch_out_inputs[i, :].unsqueeze(0)  # (1,L_y)
                        tmp_i_out_probs = torch.zeros((1, self.out_dim, self.out_max_len,)).to(device)  # (BW,D_y,L_y)
                        tmp_acc_i_log_probs = torch.zeros((1,)).to(device)  # (BW,)
                        # tmp_mean_i_log_probs=torch.zeros((1,)).to(device)   #(BW,)
                        # tmp_lens=np.zeros((beam_width,))    #(BW,)
                        aux_stacks = [[]]
                        for j in range(self.out_max_len+1):


                            # print(tmp_i_out_inputs.size(),tmp_i_encoder.size(),tmp_i_encoder.expand(tmp_i_out_inputs.size(0),-1,-1).size())
                            pred_i_out_probs = self.net.module.decoder(tmp_i_out_inputs,tmp_i_encoder.expand(tmp_i_out_inputs.size(0),-1, -1))  # (1/BW,D_y,L_y)
                            # pred_i_out_probs = self.net.module.decoding(tmp_i_out_inputs, tmp_i_encoder.expand(tmp_i_out_inputs.size(0),-1,-1))  # (1/BW,D_y,L_y)
                            tmp_i_out_probs[:, :, j] = pred_i_out_probs[:, :, j]  # (1/BW,D_y,L_y)
                            tmp_i_out_probs = tmp_i_out_probs.unsqueeze(1).expand(-1, beam_width, -1,-1)  # (1/BW,BW,D_y,L_y)
                            tmp_i_out_probs = tmp_i_out_probs.contiguous().view(-1, self.out_dim,self.out_max_len)  # (1/BW*BW,D_y,L_y)

                            pred_ij_out_log_probs = F.softmax(pred_i_out_probs[:, :, j], 1)  # (1/BW,D_y,L_y)
                            topk_ij_out_log_probs, _ = pred_ij_out_log_probs.topk(beam_width, dim=1)  # (1/BW,BW)
                            topk_ij_out_log_probs = topk_ij_out_log_probs.contiguous().view(-1)  # (1/BW*BW,)

                            tmp_acc_i_log_probs = tmp_acc_i_log_probs.unsqueeze(1).expand(-1,beam_width).contiguous().view(-1)  # (1/BW*BW,)
                            # print(tmp_acc_i_log_probs.size(),topk_ij_out_log_probs.size())
                            tmp_acc_i_log_probs = tmp_acc_i_log_probs.add(topk_ij_out_log_probs)  # (1/BW*BW,)
                            tmp_acc_i_log_probs, topk_indices = tmp_acc_i_log_probs.topk(beam_width)  # (BW,)
                            tmp_mean_i_log_probs = tmp_acc_i_log_probs / (j + 1)  # (BW,)

                            # topk_ij_out_log_probs=topk_ij_out_log_probs[topk_indices,:,:]   #(BW,
                            # print(tmp_i_out_probs.size())
                            tmp_i_out_probs = tmp_i_out_probs[topk_indices, :, :]  # (BW,D_y,L_y)

                            tmp_i_out_inputs = tmp_i_out_inputs.unsqueeze(1).expand(-1, beam_width,-1).contiguous().view(-1,self.out_max_len)  # (1/BW*BW,L_y)
                            tmp_i_out_inputs = tmp_i_out_inputs[topk_indices, :]  # (BW,L_y)
                            # print(tmp_i_out_inputs.size())
                            pred_ij_out_indices = torch.argmax(tmp_i_out_probs[:, 1:, j], dim=1) + 1  # (BW,)
                            if j < self.out_max_len:
                                tmp_i_out_inputs[:, j + 1] = pred_ij_out_indices
                            pred_ij_out_indices = pred_ij_out_indices.cpu().data.numpy()
                            # print(tmp_i_out_inputs.size())

                            # topk_indices=topk_indices
                            # print(pred_ij_out_indices)
                            # aux_stacks=np.expand_dims(np.array(aux_stacks),-1).repeat(beam_width,axis=-1).reshape(-1)  #(BW*BW,)
                            # aux_stacks=aux_stacks[topk_indices.cpu().data.numpy()].tolist()    #(BW,)
                            # if j==0:
                            #     aux_stacks=[[]]*beam_width
                            # else:
                            tmp_aux_stacks = []
                            for aux_stack in aux_stacks:
                                tmp_aux_stacks += [aux_stack] * beam_width
                            aux_stacks = [tmp_aux_stacks[topk_idx] for topk_idx in topk_indices.cpu().data.numpy()]

                            for m, (aux_stack, pred_ij_out_idx) in enumerate(
                                    zip(aux_stacks, pred_ij_out_indices.tolist())):
                                symbol = get_sym(self.out_i2w[pred_ij_out_idx])
                                aux_stacks[m].append(symbol)
                                break_flag = False
                                if j == 0 and symbol == n_symbol:  # 如果第一个就是数字
                                    break_flag = True
                                elif len(aux_stacks[m]) > 2:
                                    while len(aux_stacks[m]) > 2 and aux_stacks[m][-1] == n_symbol and aux_stacks[m][
                                        -2] == n_symbol:
                                        aux_stacks[m].pop(-1)
                                        aux_stacks[m].pop(-1)
                                        aux_stacks[m][-1] = n_symbol
                                    if len(aux_stacks[m]) == 1:
                                        break_flag = True

                                if break_flag:
                                    com_i_out_probs.append(tmp_i_out_probs[m, :, :].to('cpu').data.numpy())  # (D_y,L_y)
                                    com_mean_i_log_probs.append(tmp_mean_i_log_probs[m].to('cpu').data.numpy())  # (1,)
                                    tmp_i_out_inputs = tmp_i_out_inputs[
                                        torch.arange(tmp_i_out_inputs.size(0)) != m]  # 去掉m这个
                                    tmp_i_out_probs = tmp_i_out_probs[
                                        torch.arange(tmp_i_out_probs.size(0)) != m]  # 去掉m这个
                                    tmp_acc_i_log_probs = tmp_acc_i_log_probs[
                                        torch.arange(tmp_acc_i_log_probs.size(0)) != m]  # 去掉m这个
                                    # tmp_mean_i_log_probs=tmp_mean_i_log_probs[torch.arange(tmp_mean_i_log_probs.size(0))!=m]    #去掉m这个
                                    # beam_width-=1
                                    break
                            if tmp_i_out_probs.size(0) == 0:
                                break
                        if len(com_i_out_probs) > 0:
                            max_mean_i_log_prob_idx = com_mean_i_log_probs.index(max(com_mean_i_log_probs))  # (value)
                            # print(com_mean_i_log_probs)
                            max_com_i_out_probs = com_i_out_probs[max_mean_i_log_prob_idx][np.newaxis, :,
                                                  :]  # [(1,D_y,L_y)]
                            i_out_length = int(np.sign(np.abs(max_com_i_out_probs).sum(1)).sum(1)[0])
                            max_com_i_out_labels = np.argmax(max_com_i_out_probs[:, 1:, :], axis=1) + 1
                            max_com_i_out_labels[:, i_out_length:] = 0
                            pred_out_probs.append(max_com_i_out_probs)  # [(1,D_y,L_y)]
                            pred_out_labels.append(max_com_i_out_labels)  # [(1,D_y)]
                        else:
                            max_mean_i_log_prob_idx = torch.argmax(tmp_mean_i_log_probs)  # (tensor value)
                            max_com_i_out_probs = tmp_i_out_probs[max_mean_i_log_prob_idx, :, :].unsqueeze(0).to(
                                'cpu').data.numpy()  # [(1,D_y,L_y)]
                            i_out_length = int(np.sign(np.abs(max_com_i_out_probs).sum(1)).sum(1)[0])
                            max_com_i_out_labels = np.argmax(max_com_i_out_probs[:, 1:, :], axis=1) + 1
                            max_com_i_out_labels[:, i_out_length:] = 0
                            pred_out_probs.append(max_com_i_out_probs)  # [(1,D_y,L_y)]
                            pred_out_labels.append(max_com_i_out_labels)  # [(1,D_y)]

            pred_out_prob_np = np.concatenate(pred_out_probs, axis=0)  # (B+,D,L_y)
            # pred_out_label_np = np.concatenate(pred_out_labels, axis=0)  # (B+,L_y)
            self.net.train()

        return pred_out_np,pred_out_prob_np

class RNNSeq2Seq(TransSeq2Seq):
    def predict(self,ins,beam_width=1):
        '''
        预测样本的类别标签的接口
        :param ins: 样本特征集二维数组
        :return: 预测出的类别标签一维数组,或值
        '''
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
        if beam_width==1:
            pred_out_prob_batches = []
            # batch_pred_outs = []
            with torch.no_grad():
                for batch_features, batch_out_inputs in data_loader:
                    batch_features = batch_features.to(device)
                    batch_out_inputs = batch_out_inputs.to(device)
                    tmp_batch_outs = self.net(batch_features, batch_out_inputs, tf_rate=0)  # [B,D,L)
                    pred_out_prob_batches.append(tmp_batch_outs.to('cpu').data.numpy())  # [(B,D,L)]
                    # batch_pred_outs.append(batch_out_inputs.to('cpu').data.numpy())
            # pred_out_prob_np = np.concatenate(pred_out_prob_batches, axis=0)  # (B+,D,L)
            # pred_out_np = np.argmax(pred_out_prob_np[:, 1:, :], axis=1) + 1  # (B+,L)
            pred_out_prob_np = np.concatenate(pred_out_prob_batches, axis=0)[:, :, :-1]  # (B+,D,L)
            pred_out_np = np.argmax(pred_out_prob_np[:, 1:, :], axis=1)[:, :-1] + 1  # (B+,L)
            for i,pred_outs in enumerate(pred_out_np):
                for j,out_idx in enumerate(pred_outs):    #prefix
                    if out_idx==self.out_end_idx:
                        break
                pred_out_np[i,j:]=0
        else:
            pass
        self.net.train()
        return pred_out_np,pred_out_prob_np

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
              valid_ins=dev_features,
              valid_outs=dev_ner_labels,
              label2tag=idx2ner,
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