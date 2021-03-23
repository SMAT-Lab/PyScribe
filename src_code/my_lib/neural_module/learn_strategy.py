import torch.nn as nn

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)

class LrWarmUp(object):
    '''
    自定义Warm Up学习率变化
    '''
    def __init__(self,optimizer,min_rate=0.1,lr_decay=0.9,warm_steps=6000,reduce_steps=3000):
        self.optimizer=optimizer
        self.warm_steps=warm_steps
        self.reduce_steps=reduce_steps
        self.min_rate=min_rate
        self.lr_decay=lr_decay
        self.steps=0
        self.new_steps=0
        self.init_lrs=[param['lr'] for param in self.optimizer.param_groups]
    def step(self):
        # rate=(self.steps%self.warm_steps+1)/float(self.warm_steps)
        # for lr, param in zip(self.init_lrs, self.optimizer.param_groups):
        #     param['lr'] = lr * rate

        self.steps += 1
        if self.steps<=self.warm_steps:
            for lr,param in zip(self.init_lrs,self.optimizer.param_groups):
                param['lr']=lr*(self.steps/float(self.warm_steps))
        elif self.steps % self.reduce_steps ==0:
            for lr, param in zip(self.init_lrs, self.optimizer.param_groups):
                param['lr']=max(lr*self.min_rate,param['lr']*self.lr_decay)
                # param['lr'] = param['lr'] * self.lr_decay
    def back_step(self):
        pass
    def get_lr(self):
        return [param['lr'] for param in self.optimizer.param_groups]