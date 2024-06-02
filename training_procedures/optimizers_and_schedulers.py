import torch.optim as optim
import torch


def get_optimizer(optimizer_type, lr, model, weight_decay=0.0, betas=(0.9, 0.999)):
    if optimizer_type == 'Adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif optimizer_type == 'AdamW':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif optimizer_type == 'SGD':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'Unknown optimizer type: {optimizer_type}')
        # TODO other optimizers?
    
class InvSqrtScheduler:
    '''Inverse Square Root Scheduler'''
    def __init__(self, optimizer, warmup, maxlr, minlr=0, last_step=-1):
        self.warmup=warmup
        self.maxlr=maxlr
        self.optimizer=optimizer
        self.minlr=minlr
        if last_step==-1:
            self.current_step=0
        else:
            self.current_step=last_step
    def step(self):
        self.current_step+=1
        if self.current_step<=self.warmup:
            for p in self.optimizer.param_groups:
                p['lr']=self.maxlr*self.current_step/self.warmup
        else:
            for p in self.optimizer.param_groups:
                p['lr']=max(self.maxlr/((self.current_step/(self.warmup+1))**0.5), self.minlr)
    def state_dict(self):
        res_dict =  self.__dict__.copy()
        res_dict.pop("optimizer")
        return res_dict
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

class InvSqrtwithRestarts:
    def __init__(self, optimizer, warmup, maxlr, restart_steps, restart_lrs, minlr=0, last_step=-1):
        self.warmup=warmup
        self.maxlr=maxlr
        self.optimizer=optimizer
        self.minlr=minlr
        self.restart_steps = restart_steps
        self.restart_lrs = restart_lrs
        if last_step==-1:
            self.current_step=0
        else:
            self.current_step=last_step
        self.current_restart = 0
        self.current_restart_init_step = 0
        self.current_max_lr = self.maxlr
    def step(self):
        self.current_step+=1
        if self.current_step-self.current_restart_init_step<=self.warmup:
            for p in self.optimizer.param_groups:
                p['lr']=self.current_max_lr*(self.current_step-self.current_restart_init_step)/self.warmup
        else:
            for p in self.optimizer.param_groups:
                p['lr']=max(self.current_max_lr/(((self.current_step-self.current_restart_init_step)/(self.warmup+1))**0.5), self.minlr)
            if self.current_step == self.restart_steps[self.current_restart]:
                print('Restart at step ', self.current_step, 'current restart', self.current_restart+1)
                self.current_restart_init_step = self.current_step
                self.current_restart += 1
                self.current_restart_init_step = self.current_step
                self.current_max_lr = self.restart_lrs[self.current_restart-1]
    def state_dict(self):
        res_dict =  self.__dict__.copy()
        res_dict.pop("optimizer")
        return res_dict
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        

        
def get_scheduler(scheduler_type, maxlr, warmup, minlr, restart_steps=None, restart_lrs=None, last_step=-1, optimizer=None):
    if optimizer is None:
        return None
    if scheduler_type == 'InvSqrt':
        return InvSqrtScheduler(optimizer, warmup, maxlr, minlr, last_step)
    elif scheduler_type == 'InvSqrtwithRestarts':
        return InvSqrtwithRestarts(optimizer, warmup, maxlr, restart_steps, restart_lrs, minlr, last_step)
    else:
        raise ValueError(f'Unknown scheduler type: {scheduler_type}')
        # TODO other schedulers?