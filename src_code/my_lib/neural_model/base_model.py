#coding=utf-8
'''
所有分类器的基类，封装了模型评价方法代码以及训练和预测接口
'''
import sys
sys.path.append('../')
from my_evaluation import *
from neural_module.my_loss import CriterionNet
from balanced_data_parallel import BalancedDataParallel
import numpy as np
import pandas as pd
import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import copy
import torch.nn.functional as F

class Datasetx(Dataset):
    pass

class BaseNet(nn.Module):
    def get_init_params(self):
        if 'self' in self.init_params:
            del self.init_params['self']
        if '__class__' in self.init_params:
            del self.init_params['__class__']
        return self.init_params

class BaseModel(object):
    def __init__(self,
                 model_dir,
                 model_name='model',
                 model_id=None):
        # # os.environ['CUDA_VISIBLE_DEVICES']='0,1'
        # # gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        # # self.gpu_config=tf.ConfigProto(gpu_options=gpu_options)
        #
        # # 在开启对话Session之前，先创建一个tf.ConfigProto()实例对象
        # # 通过allow_soft_placement参数自动将无法放在GPU上的操作放回CPU
        # self.gpu_config = tf.ConfigProto()
        # # self.gpu_config.allow_soft_placement=True
        # # # self.gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.8
        # # #运行时需要多少再给多少,按需分配
        # # self.gpu_config.gpu_options.allow_growth=True
        self.model_dir=model_dir
        if self.model_dir and not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_name=model_name
        if model_id is not None:
            self.model_name=model_name+str(model_id)
        self.model_path = os.path.join(model_dir, '%s' % self.model_name)  # 模型路径

        # 配置日志信息
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # def config_params(self):
    #     '''
    #     配置模型参数的接口
    #     :return:
    #     '''
    #     pass
    def get_init_params(self):
        if 'self' in self.init_params:
            del self.init_params['self']
        if '__class__' in self.init_params:
            del self.init_params['__class__']
        return self.init_params

    def save_params(self,param_path=None):
        '''
        存储模型参数的接口
        :return:
        '''
        logging.info('Save extern parameters of %s' % self.model_name)
        if param_path is None:
            param_path = os.path.join(self.model_dir, self.model_name + '.param.pkl')
        param_dir=os.path.dirname(param_path)
        if not os.path.exists(param_dir):
            os.makedirs(param_dir)
        # if 'model_id' in kwargs:
        #     model_params_path = os.path.join(model_params_dir, '%s_%d.pkl'%(model_params_name,kwargs['model_id']))
        param_dict = {'Net':self.Net,'Dataset': self.Dataset, 'batch_size': self.batch_size}
        if 'out_begin_idx' in self.__dict__:  # 如果是seq2seq问题:
            param_dict.update({'out_begin_idx': self.out_begin_idx,
                               'in_max_len': self.in_max_len,
                               'out_max_len': self.out_max_len})
        else:
            param_dict.update({'in_max_len': self.in_max_len,
                               'out_dim': self.out_dim})
            if 'sort_unique_outs' in self.__dict__:
                param_dict.update({'sort_unique_outs':self.sort_unique_outs})
            if 'unique_outs' in self.__dict__:
                param_dict.update({'unique_outs': self.unique_outs})
        with open(param_path, 'wb') as f:
            pickle.dump(param_dict, f)



    def load_params(self,param_path=None):
        '''
        加载模型参数的接口
        :return:
        '''
        logging.info('Load extern parameters of %s' % self.model_name)
        if param_path is None:
            param_path = os.path.join(self.model_dir, self.model_name + '.param.pkl')
        with open(param_path, 'rb') as f:
            param_dict=pickle.load(f)
        self.__dict__.update(param_dict)


    def save_net(self,net_path=None):
        '''
        保存模型的接口
        :param model_dir: 模型保存的目录
        :param name: 模型名称
        :return:
        '''
        # pass
        logging.info('Save the whole model and its parameters.')
        # if net_name is None:
        #     net_name=self.model_name
        # if net_dir is None:
        #     net_dir=self.model_dir
        if net_path is None:
            net_path = os.path.join(self.model_dir, self.model_name + '.net')
        net_dir=os.path.dirname(net_path)
        if net_dir and not os.path.exists(net_dir):
            os.makedirs(net_dir)
        net_state = {'net': self.net.state_dict(),
                     'net_params':self.net.module.get_init_params()}
        # torch.save(self.net.state_dict(),model_params_path)
        torch.save(net_state,net_path)


    def load_net(self,net_path=None):
        '''
        加载模型的接口
        :param model_dir: 模型保存的目录
        :param name: 模型名称
        :return:
        '''
        if net_path is None:
            net_path = os.path.join(self.model_dir, self.model_name + '.net')
        checkpoint= torch.load(net_path)
        # net_params=dict()
        net_params=checkpoint['net_params']
        self.net=self.Net(**net_params)  #网络命名必须是Net
        self.net = nn.DataParallel(self.net)  # 并行使用多GPU，加载时放在load前面
        self.net.load_state_dict(checkpoint['net'])
        # self.net = nn.DataParallel(self.net)  # 并行使用多GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择GPU优先
        self.net.to(device)  # 数据转移到设备
        self.net.train()


    def fit(self,ins,outs):
        '''
        训练模型的接口
        :param ins: 特征集二维数组
        :param outs:  类别标签一维数组
        :return:
        '''
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.net=Net()
        pass

    def _log_fit_eval(self,
                      loss,
                      big_step, batch_step, big_epochs, batch_epochs,
                      pred_outs, true_outs,
                      seq_mode=None,
                      verbose=0):
        '''
        打印性能日志
        :param loss:
        :param big_step:
        :param batch_step:
        :param big_epochs:
        :param batch_epochs:
        :param pred_outs:
        :param true_outs:
        :param seq_mode:序列模式，'POS'或者None为普通序列分类问题（如词性标注），'NER'为序列标注问题（可能多个span label合并），
                'WHOLE'为整个序列是否全对的分类问题,'BLEU'为文本翻译评价
        :param verbose:
        :return:
        '''
        if 'avg_v2_loss_val' not in self.__dict__:
            self.avg_v2_loss_val = 0.  # 一个big epoch中平均的loss
            self.avg_v2_eval_np = np.zeros(len(self.train_metrics), dtype=np.float32)  # 一个big epoch中平均的测试结果
        if verbose == 1 and 'avg_v1_loss_val' not in self.__dict__:
            self.avg_v1_loss_val = 0.
            self.avg_v1_eval_np = np.zeros(len(self.train_metrics), dtype=np.float32)  # 100个batch epoch中平均的测试结果

        if len(pred_outs.size()) == 2 and self.out_dim > 1:  # 如果是文本分类问题
            pred_outs = torch.argmax(pred_outs, dim=1)
        elif len(pred_outs.size()) == 3 and self.out_dim > 1:  # 如果是序列标注或生成问题
            pred_outs = torch.argmax(pred_outs[:, :, :], dim=1)  # 先不要0的(B,L),必须+1，不然下标会从0开始
        if len(pred_outs.size()) == 2:  # 如果是序列标注或生成问题
            out_lens = true_outs.sign().sum(1)  # (B) 每个序列的长度
            # assert out2tag is not None
            # assert tag2span_func is not None
            # span_metric = SeqSpanMetric(out2tag, tag2span_func)  # 定义一个span_metric对象
            # 去掉每个序列的padding位，并将输出压缩成一维向量，便于下面处理
            # print(out_lens.size(0))
            # true_outs = torch.cat([true_outs[i,:out_lens[i]] for i in range(out_lens.size(0))]) #(BL-,)
            # pred_outs = torch.cat([pred_outs[i,:out_lens[i]] for i in range(out_lens.size(0))]) #(BL-,)
        true_out_np = true_outs.to('cpu').data.numpy()
        del true_outs
        pred_out_np = pred_outs.to('cpu').data.numpy()
        del pred_outs
        # print(true_out_np.shape)
        # print(pred_out_np.shape)
        # print(self.sort_unique_outs)
        if len(true_out_np.shape) == 1 and self.out_dim > 1:  # 如果输出为类别
            eval_np=np.array([metric(true_out_np, pred_out_np, unique_outs=self.sort_unique_outs)
                              for metric in self.train_metrics])
        elif len(true_out_np.shape) == 2 and self.out_dim > 1:  # 如果是序列标注问题
            if seq_mode is None or seq_mode == 'POS':
                true_out_np = np.concatenate(
                    [true_out_np[i, :out_lens[i]] for i in range(out_lens.size(0))])  # (BL-,)
                pred_out_np = np.concatenate(
                    [pred_out_np[i, :out_lens[i]] for i in range(out_lens.size(0))])  # (BL-,)
                eval_np=np.array([metric(true_out_np, pred_out_np, unique_outs=self.sort_unique_outs)
                                  for metric in self.train_metrics])
            elif seq_mode == 'NER':
                eval_np=np.array([metric(true_out_np, pred_out_np,
                                    seq_lens=out_lens,
                                    out2tag=self.out2tag,
                                    tag2span_func=self.tag2span_func)
                                  for metric in self.train_metrics])
            elif seq_mode == 'WHOLE':  # 如果为整个序列是否全对的分类问题
                true_out_list = [' '.join([str(idx) for idx in true_out_np[i, :out_lens[i]]]) for i in range(out_lens.size(0))]  # (BL-,)
                pred_out_list = [' '.join([str(idx) for idx in pred_out_np[i, :out_lens[i]]]) for i in range(out_lens.size(0))]  # (BL-,)
                eval_np = np.array([metric(true_out_list, pred_out_list) for metric in self.train_metrics])
                # print('the whole,',eval_np.shape)
            elif seq_mode == 'BLEU':
                true_out_list=[[[self.out_i2w[idx] for idx in true_out_np[i,:out_lens[i]]]] for i in range(out_lens.size(0))]  # (BL-,)
                pred_out_list = [[self.out_i2w[idx] for idx in pred_out_np[i, :out_lens[i]]] for i in range(out_lens.size(0))]  # (BL-,)
                eval_np=np.array([metric(pred_out_list,true_out_list) for metric in self.train_metrics])
                # pass  # 待定
        elif self.out_dim == 1:  # 如果输出为值
            eval_np = np.array([metric(true_out_np, pred_out_np) for metric in self.train_metrics])

        # eval_np = np.zeros(len(self.train_metrics), dtype=np.float32)
        # for k, metric in enumerate(self.train_metrics):  # 对每个评价标准
        #     if len(true_out_np.shape) == 1 and self.out_dim > 1:  # 如果输出为类别
        #         eval_np[k] = metric(true_out_np, pred_out_np, unique_outs=self.sort_unique_outs)
        #     elif len(true_out_np.shape) == 2 and self.out_dim > 1:  # 如果是序列标注问题
        #         if seq_mode is None or seq_mode == 'normal':
        #             true_out_np = np.concatenate(
        #                 [true_outs[i, :out_lens[i]] for i in range(out_lens.size(0))])  # (BL-,)
        #             pred_out_np = np.concatenate(
        #                 [pred_outs[i, :out_lens[i]] for i in range(out_lens.size(0))])  # (BL-,)
        #             eval_np[k] = metric(true_out_np, pred_out_np, unique_outs=self.sort_unique_outs)
        #         elif seq_mode == 'span':
        #             eval_np[k] = metric(true_out_np, pred_out_np,
        #                                   seq_lens=out_lens,
        #                                   out2tag=self.out2tag,
        #                                   tag2span_func=self.tag2span_func)
        #         elif seq_mode == 'whole':  # 如果为整个序列是否全对的分类问题
        #             true_out_list = [list(true_outs[i, :out_lens[i]]) for i in range(out_lens.size(0))]  # (BL-,)
        #             pred_out_list = [list(pred_outs[i, :out_lens[i]]) for i in range(out_lens.size(0))]  # (BL-,)
        #             eval_np[k] = metric(true_out_list, pred_out_list)
        #         elif seq_mode=='BLEU':
        #             pass    #待定
        #     elif self.out_dim == 1:  # 如果输出为值
        #         eval_np[k] = metric(true_out_np, pred_out_np)
        # print(self.avg_v2_eval_np.shape,eval_np.shape)
        self.avg_v2_eval_np += eval_np
        self.avg_v2_loss_val += loss.item()

        if verbose == 1:
            self.avg_v1_eval_np += eval_np
            self.avg_v1_loss_val += loss.item()

        if batch_step == batch_epochs:
            log_info = '>>>>>>>>> Big epoch step:{}/{}, train loss: {}'. \
                format(big_step, big_epochs, self.avg_v2_loss_val / batch_epochs)
            self.avg_v2_eval_np /= batch_epochs
            for avg_eval, metric in zip(self.avg_v2_eval_np, self.train_metrics):
                log_info += ', average {}: {}'.format(metric.__name__, avg_eval)
            log_info += ' <<<<<<<<<'
            logging.info(log_info)
            del self.avg_v2_loss_val
            del self.avg_v2_eval_np

        if verbose == 1 and (batch_step % 100 == 0 or batch_step == batch_epochs):
            epochs = 100 if batch_step % 100 == 0 else batch_epochs % 100
            log_info = 'Big epoch step:{}/{}, batch epoch step:{}/{}, train loss: {}'. \
                format(big_step, big_epochs, batch_step, batch_epochs, self.avg_v1_loss_val / epochs)
            self.avg_v1_eval_np /= epochs
            for avg_v1_eval, metric in zip(self.avg_v1_eval_np, self.train_metrics):
                log_info += ', average {}: {}'.format(metric.__name__, avg_v1_eval)
            logging.info(log_info)
            del self.avg_v1_loss_val
            del self.avg_v1_eval_np

        if verbose == 0:
            log_info = 'Big epoch step:{}/{}, batch epoch step:{}/{}, train Loss:{}'. \
                format(big_step, big_epochs, batch_step, batch_epochs, loss.item())  # 日志信息
            for eval_val, metric in zip(eval_np, self.train_metrics):
                log_info += ', {}: {}'.format(metric.__name__, eval_val)
            logging.info(log_info)

    def _do_validation(self, 
                       valid_ins=None, 
                       valid_outs=None, 
                       last=False,
                       increase_better=True, 
                       seq_mode=None):
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
            #首先计算loss
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择GPU优先
            pred_out_np,pred_out_prob_np=self.predict(valid_ins)   #(B,L),(B,D,L)
            pred_out_probs=torch.tensor(pred_out_prob_np)    #(B,D,L)
            if len(pred_out_probs.size())==3:    #如果是序列问题
                true_outs=[np.lib.pad(seq, (0, pred_out_probs.size(2) - len(seq)),
                                        'constant', constant_values=(0, 0)) for seq in valid_outs]
                out_lens=[np.sign(seq).sum() for seq in valid_outs]
            true_outs=torch.tensor(true_outs)
            #批量计算loss，防止cuda内存溢出
            # self.criterion.reduction='sum'
            valid_loss_val=0.
            for batch_num,i in enumerate(range(0,true_outs.size(0),self.batch_size)):
                batch_pred_out_probs=pred_out_probs[i:i+self.batch_size].to(device)
                batch_true_outs=true_outs[i:i+self.batch_size].to(device)
                valid_loss_val+=self.criterion(batch_pred_out_probs, batch_true_outs).item()
            # print(valid_loss_val,batch_num)
            valid_loss_val/=(batch_num+1e-20)
            # valid_loss_val = self.criterion(pred_out_probs, true_outs).item()
            # self.criterion.reduction='mean'
            true_outs=true_outs.data.numpy()

            #然后计算指标
            # logging.info('Previous valid loss: {}, current valid loss: {}'.format(self.valid_loss_val, valid_loss.item()))
            log_info = 'Comparison of previous and current valid loss: ({},{})'.format(self.valid_loss_val,
                                                                                       valid_loss_val)
            del pred_out_probs
            # del true_outs
            # if self.out_dim > 1:  # 如果是分类
            #     if isinstance(valid_outs[0], list) or isinstance(valid_outs[0], np.ndarray):  # 如果是序列问题:
            #         pred_outs = torch.argmax(pred_outs[:, 1:, :], dim=1) + 1  # (B,L)先不要0的(B,L),必须+1，不然下标会从0开始
            #     else:  # 如果是分类问题
            #         pred_outs = torch.argmax(pred_outs, dim=1)  # (B,)
            # if len(pred_outs.size()) == 2:  # 如果是序列标注或生成问题
            #     out_lens = true_outs.sign().sum(1)  # (B) 每个序列的长度
            #
            # true_out_np = true_outs.to('cpu').data.numpy()
            # del true_outs
            # pred_out_np = pred_outs.to('cpu').data.numpy()
            # del pred_outs

            metrics = copy.deepcopy(self.train_metrics) #如果不深拷贝，self.train_metrics下面会跟着变化
            if self.valid_metric is not None:  # 如果有valid metric
                metrics.append(self.valid_metric)
            if len(pred_out_np.shape) == 1 and self.out_dim > 1:  # 如果为文本分类
                valid_evals=[metric(true_outs, pred_out_np, unique_outs=self.sort_unique_outs) for metric in metrics]
            elif len(pred_out_np.shape) == 2 and self.out_dim > 1:  # 如果为序列标注或生成
                if seq_mode is None or seq_mode == 'POS':
                    pred_outs = np.concatenate([pred_out_np[i,:out_len] for i,out_len in enumerate(out_lens)])  # (BL-,)
                    true_outs=np.concatenate([true_outs[i][:out_len] for i,out_len in enumerate(out_lens)])
                    valid_evals=[metric(pred_outs, true_outs, unique_outs=self.sort_unique_outs) for metric in metrics]
                elif seq_mode == 'NER':
                    valid_evals=[metric(true_outs, pred_out_np,
                                              seq_lens=out_lens,
                                              out2tag=self.out2tag,
                                              tag2span_func=self.tag2span_func)
                                 for metric in metrics]
                elif seq_mode == 'WHOLE':  # 如果为整个序列是否全对的分类问题(如MWP）
                    # true_out_list = [list(valid_outs[i][:out_len]) for i,out_len in enumerate(out_lens)]  # (BL-,)
                    # pred_out_list = [list(pred_out_np[i, :out_ln]) for i,out_len in enumerate(out_lens)]  # (BL-,)
                    true_out_list = [' '.join([str(idx) for idx in true_out]) for true_out in true_outs]  # (BL-,)
                    pred_out_list = [' '.join([str(idx) for idx in pred_out]) for pred_out in pred_out_np]  # (BL-,)
                    valid_evals=[metric(true_out_list, pred_out_list) for metric in metrics]
                elif seq_mode == 'BLEU':
                    true_out_list = [[[self.out_i2w[idx] for idx in (true_out[:true_out.tolist().index(0)]
                                       if 0 in true_out else true_out)]] for true_out in true_outs]  # (BL-,)
                    pred_out_list = [[self.out_i2w[idx] for idx in (pred_out[:pred_out.tolist().index(0)]
                                        if 0 in pred_out else pred_out)] for pred_out in pred_out_np]

                    valid_evals = [metric(pred_out_list, true_out_list) for metric in metrics]
            elif self.out_dim == 1:  # 如果输出为值
                valid_evals=[metric(true_outs, pred_out_np) for metric in metrics]

            # valid_evals = []
            # for metric in metrics:  # 对每个评价标准
            #     if len(pred_out_np.shape) == 1 and self.out_dim > 1:  # 如果为文本分类
            #         valid_evals.append(metric(valid_outs, pred_out_np, unique_outs=self.sort_unique_outs))
            #     elif len(pred_out_np.shape) == 2 and self.out_dim > 1:  # 如果为序列标注或生成
            #
            #         if seq_mode is None or seq_mode == 'normal':
            #             pred_outs = np.concatenate([pred_out_np[i,:out_len] for i,out_len in enumerate(out_lens)])  # (BL-,)
            #             true_outs=np.concatenate([valid_outs[i][:out_len] for i,out_len in enumerate(out_lens)])
            #             valid_evals.append(metric(pred_outs, true_outs, unique_outs=self.sort_unique_outs))
            #         elif seq_mode == 'span':
            #             valid_evals.append(metric(valid_outs, pred_out_np,
            #                                       seq_lens=out_lens,
            #                                       out2tag=self.out2tag,
            #                                       tag2span_func=self.tag2span_func))
            #         elif seq_mode == 'whole':  # 如果为整个序列是否全对的分类问题
            #             true_out_list = [list(valid_outs[i][:out_len]) for i,out_len in enumerate(out_lens)]  # (BL-,)
            #             pred_out_list = [list(pred_out_np[i, :out_len]) for i,out_len in enumerate(out_lens)]  # (BL-,)
            #             valid_evals.append(metric(true_out_list, pred_out_list))
            #         elif seq_mode == 'BLEU':
            #             pass  # 待定
            #     elif self.out_dim == 1:  # 如果输出为值
            #         valid_evals.append(metric(valid_outs, pred_out_np))

            for i, metric in enumerate(metrics):
                log_info += ', average {}: ({},{})'.format(metric.__name__, self.valid_evals[i], valid_evals[i])

            logging.info(log_info)
            is_better = False
            if self.valid_metric is not None:  # 如果有valid metric
                if increase_better and valid_evals[-1] >= self.valid_evals[-1]:\
                    is_better = True
                elif not increase_better and valid_evals[-1] <= self.valid_evals[-1]:\
                    is_better = True
            elif valid_loss_val <= self.valid_loss_val:
                is_better = True
            if is_better:  # 如果表现更好，暂存步数
                # self.best_net = copy.deepcopy(self.net)
                # self.best_net = self.net.clone()
                self.valid_loss_val = valid_loss_val
                self.valid_evals = valid_evals
            #     self.new_lr_steps=self.scheduler.steps
                torch.save(self.net.state_dict(),os.path.join(self.model_dir,self.model_name+'_best_net.net'))
            # else:
            #     # self.net.load_state_dict(torch.load(self.model_dir+'_best_net.net'))
            #     # self.net.train()
            #     self.scheduler.steps=self.new_lr_steps+random.randrange(-back_steps,back_steps+1)
            #     self.scheduler.step()

            # self.net=copy.deepcopy(self.best_net)
            # self.net.in()
            # print(is_better, self.scheduler.steps)
        elif last:
            self.net.load_state_dict(torch.load(os.path.join(self.model_dir,self.model_name+'_best_net.net')))
            self.net.train()
        # elif last and 'best_net' in self.__dict__:
        #     # print('The last!')
        #     # self.net = copy.deepcopy(self.best_net)
        #     self.net = self.best_net.clone()
        #     del self.best_net
        # self.net.train()

    def predict(self,ins):
        '''
        预测样本的类别标签的接口
        :param ins: 样本特征集二维数组
        :return: 预测出的类别标签一维数组,或值
        '''
        logging.info('Predict outputs of %s' % self.model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择GPU优先
        self.net=self.net.to(device)   # 数据转移到设备,不重新赋值不行
        self.net.eval()
        if 'out_begin_idx' in self.__dict__:  #如果是标准的seq2seq问题，有起始id和终止id，无beam search
            dataset = self.Dataset(ins,
                                   in_max_len=self.in_max_len,
                                   out_max_len=self.out_max_len,
                                   out_begin_idx=self.out_begin_idx)
            data_loader = DataLoader(dataset=dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                            num_workers=8)
            pred_out_prob_batches=[]
            # batch_pred_outs = []
            with torch.no_grad():
                for batch_features,batch_out_inputs in data_loader:
                    batch_features = batch_features.to(device)
                    batch_out_inputs=batch_out_inputs.to(device)
                    tmp_batch_outs=[]
                    for i in range(self.out_max_len+1): #因为有终止id
                        pred_outs=self.net(batch_features,batch_out_inputs) #(B,D,L)
                        tmp_batch_outs.append(pred_outs[:,:,i].unsqueeze(-1).to('cpu').data.numpy())  #(B,D,1)
                        if i<self.out_max_len:
                            batch_out_inputs[:,i+1]=torch.argmax(pred_outs[:,1:,i],dim=1)+1   #必须+1,(B,L)
                    #     del pred_outs
                    # del batch_features
                    # del batch_out_inputs
                        # else:
                        #     batch_out_inputs[:,:-1]=batch_out_inputs[:,1:]
                        #     batch_out_inputs[:,-1]=torch.argmax(pred_out_probs[:,1:,i],dim=-1)+1 #必须+1,(B,L)
                    pred_out_prob_batches.append(np.concatenate(tmp_batch_outs,axis=-1))   #[(B,D,L)]
                    # batch_pred_outs.append(batch_out_inputs.to('cpu').data.numpy())

            pred_out_prob_np=np.concatenate(pred_out_prob_batches,axis=0)[:,:,:-1] #(B+,D,L)
            pred_out_np=np.argmax(pred_out_prob_np[:,1:,:],axis=1)[:,:-1]+1    #(B+,L)

        else:    #如果是普通的文本分类、回归或序列标注等问题
            dataset = self.Dataset(ins, in_max_len=self.in_max_len)
            data_loader = DataLoader(dataset=dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                  num_workers=8)
            # batch_pred_outs=[]
            # zero_masks=[]
            # with torch.no_grad():
            #     for batch_features in data_loader:
            #         batch_features=batch_features.to(device)
            #         pred_outs = self.net(batch_features)    #默认是文本回归(B,*,*)
            #         batch_pred_outs.append(pred_outs)
            # pred_outs=torch.cat(batch_pred_outs,dim=0)   #(B+,*,*),默认是文本回归
            # if len(pred_outs.size()) == 2 and self.out_dim > 1:  # 如果是文本分类
            #     pred_outs = torch.argmax(pred_outs, dim=1)
            # elif len(pred_outs.size()) == 3 and self.out_dim > 1:  # 如果是序列标注问题
            #     pred_outs = torch.argmax(pred_outs[:, 1:, :], dim=1) + 1  # 先不要0的(B,L),必须+1，不然下标会从0开始
            #     zero_mask = batch_features.eq(0)  # (B,L)
            #     pred_outs = pred_outs.masked_fill(zero_mask, 0)  # (B,L)
            pred_out_prob_batches=[]
            pred_out_batches = []
            with torch.no_grad():
                for batch_features in data_loader:
                    batch_features=batch_features.to(device)
                    pred_out_probs = self.net(batch_features)    #默认是文本回归
                    pred_out_prob_batches.append(pred_out_probs.to('cpu').data.numpy())    #默认回归问题的情况下，规定输出值与概率是一回事,
                    # ins=torch.cat()
                    if len(pred_out_probs.size())==2 and self.out_dim>1:    #如果是文本分类
                        pred_outs = torch.argmax(pred_out_probs, dim=1)
                    elif len(pred_out_probs.size())==3 and self.out_dim>1:  #如果是序列标注问题
                        pred_outs=torch.argmax(pred_out_probs[:,1:,:],dim=1)+1 #先不要0的(B,L),必须+1，不然下标会从0开始
                        zero_mask = batch_features.eq(0)  # (B,L)
                        if len(batch_features.size()) == 3:
                            zero_mask=zero_mask[:,0,:]
                        pred_outs = pred_outs.masked_fill(zero_mask, 0)  # (B,L)
                    pred_out_batches.append(pred_outs.to('cpu').data.numpy())
            pred_out_prob_np =np.concatenate(pred_out_prob_batches, axis=0)
            pred_out_np=np.concatenate(pred_out_batches,axis=0)
        self.net.train()
        return pred_out_np,pred_out_prob_np #序列概率输出形状为（B,D,L)


    def eval_class(self,
                   test_ins,
                   test_outs,
                   unique_outs=None,
                   focus_labels=[],
                   test_metrics=[get_sensitivity_series,
                                 get_specificity_series,
                                 get_balanced_accuracy_series,
                                 ],
                   percentage=False
                   ):
        '''
        普通分类的模型评价
        :param test_ins: 测试样本特征集二维数组
        :param test_outs: 类别标签一维数组
        :param unique_outs: 不同种类的标记列表
        :param test_metrics: 评价方法列表
        :param focus_labels: 关注类别列表
        :return:
        '''
        logging.info('Evaluate %s' % self.model_name)
        pred_outs,_ = self.predict(test_ins)  # 预测出的标记一维数组
        assert self.out_dim > 1 and len(pred_outs.shape) == 1  # 根据输出维度和size断言为文本分类问题

        if unique_outs is None and 'sort_unique_outs' in self.__dict__:
            # 如果unique_labels参数为None，但类里已经有了sort_unique_labels属性，还是用sort_unique_labels
            unique_outs = self.sort_unique_outs
        elif unique_outs is None and 'unique_outs' in self.__dict__:
            # 否则，如果unique_labels参数为None，但类里已经有了unique_labels属性
            unique_outs = sorted(self.unique_outs)
        # 填充代码
        if focus_labels == []:  # 不指定关注类别的话，默认关注所有类别(focus_labels为[])
            focus_labels = sorted(np.unique(test_outs))  # 不同标记
        index = focus_labels  # 索引
        columns = [metric.__name__ for metric in test_metrics]  # 列名
        eval_df = pd.DataFrame(data=np.empty(shape=(len(index), len(columns))),
                               index=index,
                               columns=columns,
                               )  # 评价结果
        # print(eval_df.shape)
        for metric in test_metrics:
            # 对每种评价
            eval_result = metric(test_outs, pred_outs, unique_outs=unique_outs)  # 计算评价结果
            if isinstance(eval_result, float) or isinstance(eval_result, int):
                # if eval_result.digit():
                # 如果评价结果是一个数字，该结果放在该列的第一行位置，该列其他元素置None，并将该列放到最后一列
                eval_df.loc[:, metric.__name__] = None  # 该列置None
                eval_df.loc[:, metric.__name__].iloc[0] = eval_result  # 结果放在该列的第一个位置，列用名，行是索引，只能这么做
                tmp_series = eval_df.loc[:, metric.__name__]  # 取出该列
                eval_df.drop(labels=[metric.__name__], axis=1, inplace=True)  # 删除该列
                eval_df = pd.concat((eval_df, tmp_series), axis=1)  # 将该列拼接到最后一列
            elif isinstance(eval_result, pd.Series):  # 如果评价结果是一个Series
                eval_df.loc[focus_labels, metric.__name__] = eval_result[focus_labels].values
            # print(eval_df.loc[focus_labels,metric.__name__])
            # print(eval_df)

        return eval_df

    def eval_seq(self,
                 test_ins,
                 test_outs,
                 test_metrics=[get_span_micro_F1],
                 seq_mode=None
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
        logging.info('Evaluate %s' % self.model_name)
        pred_outs,_ = self.predict(test_ins)  # 预测出的标记二维数组
        out_lens = [np.sign(seq).sum() for seq in test_outs]
        assert self.out_dim > 1 and len(pred_outs.shape) == 2  # 根据输出维度和size断言为序列标注问题
        test_outs = [np.lib.pad(seq, (0, pred_outs.shape[1] - len(seq)),
                                'constant', constant_values=(0, 0)) for seq in test_outs]
        # span_metric=SeqSpanMetric(out2tag,tag2span_func)  #定义一个span_metric对象

        # if unique_outs is None and 'sort_unique_outs' in self.__dict__:
        #     # 如果unique_labels参数为None，但类里已经有了sort_unique_labels属性，还是用sort_unique_labels
        #     unique_outs = self.sort_unique_outs
        # elif unique_outs is None and 'unique_outs' in self.__dict__:
        #     # 否则，如果unique_labels参数为None，但类里已经有了unique_labels属性
        #     unique_outs = sorted(self.unique_outs)

        # 填充代码
        # if focus_spans == []:  # 不指定关注span类别的话，默认关注所有类别(focus_labels为[])
        #     focus_spans = sorted(np.unique(test_outs))  # 不同标记
        eval_dic = dict()
        for metric in test_metrics:
            if seq_mode is None or seq_mode == 'POS':
                # print(metric.__name__)
                pred_out_list = np.concatenate([pred_outs[i, :out_len] for i, out_len in enumerate(out_lens)])  # (BL-,)
                true_out_list = np.concatenate([test_outs[i][:out_len] for i, out_len in enumerate(out_lens)])
                eval_result = metric(pred_out_list, true_out_list, unique_outs=self.sort_unique_outs)
            elif seq_mode == 'NER':
                eval_result = metric(true_labels=test_outs,
                                     pred_labels=pred_outs,
                                     seq_lens=out_lens,
                                     out2tag=self.out2tag,
                                     tag2span_func=self.tag2span_func)
            elif seq_mode == 'WHOLE':  # 如果为整个序列是否全对的分类问题(如MWP）
                true_out_list = [list(test_out) for test_out in test_outs]  # [(B,L),...]
                pred_out_list = [list(pred_out) for pred_out in pred_outs]  # [(B,L),...]
                eval_result = metric(true_out_list, pred_out_list)
            elif seq_mode == 'BLEU':
                true_out_list = [[[self.out_i2w[idx] for idx in (test_out[:test_out.tolist().index(0)]
                                    if 0 in test_out else test_out)]] for test_out in test_outs]  # (BL-,)
                pred_out_list = [[self.out_i2w[idx] for idx in (pred_out[:pred_out.tolist().index(0)]
                                if 0 in pred_out else pred_out)] for pred_out in pred_outs]
                eval_result = metric(pred_out_list,true_out_list)
            eval_dic[metric.__name__] = dict()
            if isinstance(eval_result, float) or isinstance(eval_result, int):
                eval_dic[metric.__name__]['OVERALL'] = eval_result
            elif isinstance(eval_result, pd.Series):  # 如果评价结果是一个Series
                eval_dic[metric.__name__] = dict(eval_result)
        eval_df = pd.DataFrame(eval_dic)
        return eval_df

    def eval_reg(self,
                   test_ins,
                   test_outs,
                   test_metrics=[get_pearson_corr_val,
                                 get_spearman_corr_val,
                                 get_kendall_corr_val,
                                 ]
                   ):
        '''
        回归的模型评价
        :param test_ins: 测试样本特征集二维数组
        :param test_outs: 类别标签一维数组
        :param unique_outs: 不同种类的标记列表
        :param test_metrics: 评价方法列表
        :param focus_labels: 关注类别列表
        :return:
        '''
        logging.info('Evaluate %s' % self.model_name)
        pred_outs,_  = self.predict(test_ins)  # 预测出的标记一维数组
        assert len(pred_outs.shape) == 1 and self.out_dim == 1  # 如果输出为值
        columns = [metric.__name__ for metric in test_metrics]  # 列名
        eval_df = pd.DataFrame(data=np.empty(shape=(1, len(columns))),
                               columns=columns,
                               )  # 评价结果
        for metric in test_metrics:
            # 对每种评价
            eval_result = metric(list(test_outs), list(pred_outs))  # 计算评价结果
            eval_df.loc[0, metric.__name__] = eval_result
            # print('test_outs:\n',test_outs[:100])
            # print('pred_outs:\n',pred_outs[:100])
        return eval_df


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