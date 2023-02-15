import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

### Do not modif the following codes
class ctx_noparamgrad(object):
    def __init__(self, module):
        self.prev_grad_state = get_param_grad_state(module)
        self.module = module
        set_param_grad_off(module)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        set_param_grad_state(self.module, self.prev_grad_state)
        return False
        
def get_param_grad_state(module):
    return {param: param.requires_grad for param in module.parameters()}

def set_param_grad_off(module):
    for param in module.parameters():
        param.requires_grad = False

def set_param_grad_state(module, grad_state):
    for param in module.parameters():
        param.requires_grad = grad_state[param]
### Ends


### PGD Attack
class PGDAttack():
    def __init__(self, attack_step=10, eps=8 / 255, alpha=2 / 255, loss_type='ce', targeted=True, 
                 num_classes=10):
        '''
        attack_step: number of PGD iterations
        eps: attack budget
        alpha: PGD attack step size
        '''
        ### Your code here
        self.attack_step = attack_step
        self.eps = eps
        self.alpha = alpha
        self.loss_type = loss_type
        self.targeted = targeted
        self.num_classes = num_classes
        self.loss = None
        ### Your code ends

    def ce_loss(self, logits, y):
        ### Your code here
        loss = nn.CrossEntropyLoss()
        loss_output = loss(logits, y)
        loss_output.backward()
        ### Your code ends

    def cw_loss(self, logits, y):
        ### Your code here
        if self.targeted==False:
            Zt0 = logits[[i for i in range(logits.size()[0])], y]
            top2Zc, top2ZcIndex = torch.topk(logits, dim=-1, k=2)
            mask = torch.where(top2ZcIndex[:, 0] == y, 1, 0)
            maxZc = top2Zc[[i for i in range(top2Zc.size()[0])], mask]
            tau = 0
            loss = torch.clamp(Zt0 - maxZc, min=-tau)
            loss = torch.sum(loss)
            loss.backward()
        else:
            top2Zc, top2ZcIndex = torch.topk(logits, dim=-1, k=2)
            mask = torch.where(top2ZcIndex[:, 0] == 1, 1, 0)
            maxZc = top2Zc[[i for i in range(top2Zc.size()[0])], mask]
            Zt = logits[:, 1]
            tau = 0
            loss = torch.clamp(maxZc - Zt, min=-tau)
            loss = torch.sum(loss)
            loss.backward()
        ### Your code ends

    def perturb(self, model: nn.Module, X, y):
        delta = torch.zeros_like(X)
        
        ### Your code here
        X_var = X.data
        X.requires_grad = True
        for i in range(0, self.attack_step):
            if X.grad is not None:
                X.grad.zero_()
            output = model(X)
            model.zero_grad()
            if (self.loss_type=='ce'):
                self.ce_loss(output, y)
            else:
                self.cw_loss(output, y)
            data_grad = X.grad.data
            sign_data_grad = data_grad.sign()
            temp_delta = self.alpha * sign_data_grad
            temp_img = X + temp_delta 
            delta = temp_img - X_var #get gradient across all iterations
            delta = torch.clamp(delta, -self.eps, self.eps)
            X.data = torch.clamp(X.data + delta, 0, 1)
        ### Your code ends
        #X.data = X_var
        #return delta


### FGSMAttack
'''
Technically you can transform your PGDAttack to FGSM Attack by controling parameters like `attack_step`. 
If you do that, you do not need to implement FGSM in this class.
'''
class FGSMAttack():
    def __init__(self, eps=8 / 255):
        self.eps = eps

    def perturb(self, model: nn.Module, X, y):
        delta = torch.ones_like(X)
        ### Your code here
        X.requires_grad = True
        output = model(X)
        loss = nn.CrossEntropyLoss()
        loss_output = loss(output, y)
        model.zero_grad()
        loss_output.backward()
        data_grad = X.grad.data
        sign_data_grad = data_grad.sign()
        delta = sign_data_grad * self.eps
        delta = torch.clamp(delta, -self.eps, self.eps)
        ### Your code ends
        return delta
