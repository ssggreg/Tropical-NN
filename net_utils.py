import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import pandas as pd
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.optim.optimizer import Optimizer
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import math    
    
class one_h_Net(nn.Module):
    def __init__(self,h):
        super(type(self), self).__init__()
        self.f1 = nn.Linear(2, h)
        self.f2 = nn.Linear(h,  1)
        #self.f3 = nn.Linear(8,  8)
        #self.f4 = nn.Linear(8,  8)
        #self.f3 = nn.Linear(4,  1)
    def forward(self, x):
        out = x
        out = F.relu(self.f1(out))
        #out = F.relu(self.f2(out))
        #out = F.relu(self.f3(out))
        #out = F.relu(self.f4(out))
        out =self.f2(out)
        return out
    
class two_h_Net(nn.Module):
    def __init__(self,h):
        super(type(self), self).__init__()
        self.f1 = nn.Linear(2, h)
        self.f2 = nn.Linear(h, h)
        self.f3 = nn.Linear(h,  1)
        #self.f3 = nn.Linear(8,  8)
        #self.f4 = nn.Linear(8,  8)
        #self.f3 = nn.Linear(4,  1)
    def forward(self, x):
        out = x
        out = F.relu(self.f1(out))
        out = F.relu(self.f2(out))
        #out = F.relu(self.f2(out))
        #out = F.relu(self.f3(out))
        #out = F.relu(self.f4(out))
        out =self.f3(out)
        return out    
    
class Maxout(nn.Module):

    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)


    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        out = out.view(*shape)
        m, i = out.max(max_dim)
        return m
    
class maxout_Net(nn.Module):
    def __init__(self,h,n):
        super(type(self), self).__init__()
        self.f1 = Maxout(2, h, n)
        self.f2 = Maxout(h,  2, n)
        #self.f3 = nn.Linear(8,  8)
        #self.f4 = nn.Linear(8,  8)
        #self.f3 = nn.Linear(4,  1)
    def forward(self, x):
        out = x
        out = self.f1(out)
        out =self.f2(out)
        return (out[:,0]-out[:,1]).view(-1,1)
    
    
class csvDataset():
    def __init__(self, data,label, transform=None):
        self.label = label
        self.data = data
        #self.train_set = TensorDataset()
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        sample = { 'data': data,'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class ToTensor(object):
    def __call__(self, sample):
        data, label= sample['data'],sample['label']
        return {'data': torch.from_numpy(data).float(),'label': torch.from_numpy(label).float()}
    
    
def accuracy(model, dataloader,size):
    """ Computes the model's classification accuracy on the train dataset
    Computes classification accuracy and loss(optional) on the test dataset
    The model should return logits
    """
    model.eval()
    with torch.no_grad():
        correct = 0.
        for i in (dataloader):
            inputs = i['data']
            labels = i['label']
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            pred = (outputs>0).float()
            correct += (pred.view(-1,1) == labels).sum().item()
        accuracy = correct / size
        
    return accuracy

class Adam_bis(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, prec = 2):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,prec = prec)
        super(Adam_bis, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_bis, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data = (p.data.addcdiv_(-step_size, exp_avg, denom)).mul(10**group['prec']).trunc().div(10**group['prec'])

        return loss
    
def train_model_cc_fast(model, trainloader,inference_loader, criterion, optimizer,size, num_epochs=25):
    model = model.cuda()
    train_accuracy = []
    eval_n=[]
    weights=[]
    for epoch in range(num_epochs):
        model.train(True)
        for i in trainloader:
            inputs = i['data']
            labels = i['label']
            inputs, labels = inputs.cuda(), labels.cuda()
            def closure():
                optimizer.zero_grad()
                logits = model.forward(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                return loss
            loss = optimizer.step(closure)

        train_accuracy.append(accuracy(model, trainloader,size))
        eval_n.append(eval_net(inference_loader,model))
        weights.append(get_weights(model))
        
    return model, train_accuracy,eval_n,weights    
    
def eval_net(inference_loader,model):

    model.eval()
    with torch.no_grad():
        for i in inference_loader:
                inputs = i['data']
                labels = i['label']
                inputs, labels = inputs.cuda(), labels.cuda()
        out = model.forward(inputs)
    
    return ((out>-0.2)*(out<0.2)).cpu().numpy()

def decompose(w):
        return (w+abs(w))/2,(-w+abs(w))/2
    
def get_weights(model):
    rez={}
    for name, param in model.named_parameters():
        if param.requires_grad:
            rez[name] = param.cpu().detach().numpy()
    return rez

class Polynomial_Tropical_net:
    
    def __init__(self, poids,prec):

        self.poids = poids
        self.size = int(len(poids.values())/2)
        self.F=[]
        self.G=[]
        self.H=[]
        self.prec = prec
        self.shapes=[2]
        self.hyp = []
        for i in range(1,self.size+1):
            W = self.poids['f'+str(i)+'.weight']
            self.shapes.append(W.shape[0])
        
    #def __call__(self, x):  
     #   res = -1e+8
      #  for coeff in self.coefficients:
       #     a,b,c = coeff
        #    res = np.maximum(c + x[0]* a+x[1]* b,res)
        #return res
    def get_hypersurface(self,T):
        for i in range(T.shape[0]):
            self.generate_polynom(T[i])
            
    def trace_hypersurface(self):
        R = np.zeros((len(self.hyp),2))
        for i in range(len(self.hyp)):
            R[i]=self.hyp[i]
        plt.scatter(R[:, 0], R[:, 1], marker='o',s=25, edgecolor='k')       

    def generate_polynom(self,x):
        
        state_f = x
        state_g = np.zeros((self.shapes[0]))
        state_h = np.zeros((self.shapes[0]))
        self.F.append(state_f)
        self.G.append(state_g)
        self.H.append(state_h)
        
        
        for i in range(1,self.size):
            W = self.poids['f'+str(i)+'.weight']
            W_plus,W_moins = decompose(W)
            b = self.poids['f'+str(i)+'.bias']
            
            #a = np.zeros((W.shape[0]))
            #c = np.zeros((W.shape[0]))
            #for k in range(W.shape[0]):
                
             #       print(W[0].shape)
                    
              #      a[k]=np.maximum(W_plus[k,:].dot(state)+b[k],0+W_moins[k,:].dot(state))
               #     b[k]=W_moins[k,:].dot(state)
            state_g_next = W_plus.dot(state_g) + W_moins.dot(state_f)
            state_h_next = W_plus.dot(state_f) + W_moins.dot(state_g)+b
            state_f_next = np.maximum(state_h_next,state_g_next+0)
            
            if (min(abs((state_h_next-state_g_next)))<self.prec):
                #print(x,'added')
                self.hyp.append(x)
            
            #print(W_plus)
            #print(state_f_next.shape,'f')
            #print(state_g_next.shape,'g')
            #print((state_h_next+b).shape,'h')
            #print((state_f_next+b-state_g_next,'out'))

            state_g,state_h,state_f=state_g_next,state_h_next,state_f_next
            
            self.F.append(state_f)
            self.G.append(state_g)
            self.H.append(state_h)
            
            #print(i , 'layer done')
            
            
            
        W = self.poids['f'+str(self.size)+'.weight']
        W_plus,W_moins = decompose(W)
        b = self.poids['f'+str(self.size)+'.bias']
        state_g_next = W_plus.dot(state_g) + W_moins.dot(state_f)
        state_h_next = W_plus.dot(state_f) + W_moins.dot(state_g)
        
        ll = state_h_next+b-state_g_next
        #print(self.size , 'layer done')
        return ll
            