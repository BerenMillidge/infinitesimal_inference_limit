# code that implements PCNs and BP ANNs includes core base classes
import torch.nn.functional as F
import torch.autograd.functional as taf
from copy import deepcopy
import torch.nn.functional as F
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

def accuracy(out, labels):
    with torch.no_grad():
        maxes = torch.argmax(out.detach(), dim=1)
        corrects = maxes == labels
        return torch.sum(corrects).item() / len(corrects) 



class NN_model(nn.Module):
    def __init__(self,input_size, hidden_sizes, output_size, batch_size):
        super(NN_model, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.batch_size = batch_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inp):
        x = torch.relu(self.fc1(inp))
        x = torch.relu(self.fc2(x))
        #out = self.logsoftmax(self.fc3(x))
        out = self.fc3(x)
        return out
    
    def batch_accuracy(self, preds, labels):
        pred_idxs = torch.argmax(preds,dim=0)
        corrects = pred_idxs == labels
        return torch.sum(corrects).item() / self.batch_size

    def compute_test_accuracy(self, testset):
        N = 0
        total_acc = 0
        for i, (images, labels) in enumerate(testset):
            images = images.view(images.shape[0], -1)
            #onehot_labels = self.onehot_batch(labels).permute(1,0).float()
            out = self.forward(images)
            acc = accuracy(out, labels)
            N += 1
            total_acc += acc
        return total_acc / N

def linear_base_function(input, weights, biases, **kwargs):
    f = kwargs["f"]
    return f(F.linear(input, weights, biases))


class PCLayer2_GPU(object):
    def __init__(self, device, base_fn,params, **kwargs):
        self.device = device
        self.base_fn = base_fn
        self.params = params
        # set to correct device
        self.set_device(self.device)
        self.kwargs = kwargs 

    def set_device(self, device):
        self.device = device
        for i in range(len(self.params)):
            self.params[i] = self.params[i].to(self.device)

    def mlp_forward(self, x):
        weights, biases = self.params
        self.x = x.clone()
        return self.f(self.base_fn(x, weights,biases))

    def forward(self, x):
        x = x.to(self.device)
        self.x = x.clone()
        return self.base_fn(x, *self.params, **self.kwargs)

    def backward(self, e):
        back_fn = lambda x,*p: self.base_fn(x, *p, **self.kwargs) 
        out, grads =  taf.vjp(back_fn, tuple([self.x] + self.params), e)
        self.dx = grads[0]#.to(self.device)
        self.dparams = grads[1:]#.to(self.device)
        return self.dx, self.dparams

    def update_params(self, lr):
        for i in range(len(self.params)):
            self.params[i] = self.params[i] - (lr * self.dparams[i]) 
        return self.dparams

    def set_params(self):
        self.params = [nn.Parameter(p) for p in self.params]

    def unset_params(self):
        self.params = [p.detach() for p in self.params]

class PC_Net2_GPU(object):
    def __init__(self, pc_layers, batch_size, mu_dt, lr, N_fp_steps, N_reconf_steps,use_reconfiguration = False, use_backprop = False, use_fp_reconfiguration = False, clamp_val = 10, fp_lr = 1, device="cpu"):
        self.pc_layers = pc_layers
        self.batch_size = batch_size
        self.mu_dt = mu_dt
        self.lr = lr
        self.N_fp_steps = N_fp_steps
        self.N_reconf_steps = N_reconf_steps
        self.clamp_val = clamp_val
        self.use_reconfiguration = use_reconfiguration
        self.use_backprop = use_backprop
        self.use_fp_reconfiguration = use_fp_reconfiguration
        self.device = device
        self.fp_lr = fp_lr
        for l in self.pc_layers:
            l.set_device(self.device)


    def forward(self, inp):
        with torch.no_grad():
            x = inp.clone().to(self.device)
            for l in self.pc_layers:
                l.x = deepcopy(x)
                x = l.forward(x)
            return x

    def set_params(self):
        for l in self.pc_layers:
            l.set_params()

    def unset_params(self):
        for l in self.pc_layers:
            l.unset_params()

    def batch_accuracy(self, preds, labels):
        pred_idxs = torch.argmax(preds,dim=0)
        corrects = pred_idxs == labels
        return torch.sum(corrects).item() / self.batch_size

    def onehot_batch(self, ls):
        return F.one_hot(ls, 10).permute(1,0)

    def backprop_infer(self, inp, label, loss_fn,loss_fn_str = "mse"):
        inp = inp.to(self.device)
        label = label.to(self.device)
        out = self.forward(inp)
        es = [[] for i in range(len(self.pc_layers)+1)]
        loss, dl = taf.vjp(lambda out: loss_fn(out, label),out,torch.tensor(1).to(self.device))
        es[-1] = deepcopy(dl)
        dws = [[] for i in range(len(self.pc_layers))]
        for i in reversed(range(len(self.pc_layers))):
            dx, dparams = self.pc_layers[i].backward(es[i+1])
            es[i] = deepcopy(dx)
            dws[i] = deepcopy(dparams)
        self.dws = dws
        return es, dws

    def compare_bp_autograd(self, inp, label, loss_fn = "mse"):

        inp = torch.tensor(inp.reshape(784,self.batch_size),dtype=torch.float).to(self.device)
        label = label.to(self.device)
        if loss_fn == "mse":
            label = torch.tensor(self.onehot_batch(label).reshape(10,self.batch_size), dtype=torch.float)
        es, dws = self.backprop_infer(inp, label,loss_fn_str = loss_fn)
        # begin setup for autograd
        self.set_params()
        out = self.forward(inp)
        if loss_fn == "mse":
            loss_fn = nn.MSELoss(reduction = "sum")
        if loss_fn == "crossentropy":
            loss_fn = nn.CrossEntropyLoss(reduction="sum")
        out = out.T
        
        loss = loss_fn(out, label)
        print("BP LOSS: ", loss.item())
        loss.backward()
        dwss = []
        paramss = []
        # iterate over all param list
        for i,l in enumerate(self.pc_layers):
            for j in range(len(dws[i])):
                print("DW")
                print(dws[i][j])
                print("GRAD")
                print(l.params[j].grad)
                dwss.append(deepcopy(dws[i][j]))
                paramss.append(deepcopy(l.params[j].grad.detach()))
        self.unset_params()
        return dwss, paramss

    def compare_bp_reconf(self, inp, label, loss_fn = "mse"):
        inp = torch.tensor(inp.reshape(784,self.batch_size),dtype=torch.float).to(self.device)
        label = label.to(self.device)
    #img = torch.tensor(img,dtype=torch.float)
        if loss_fn == "mse":
            label = torch.tensor(self.onehot_batch(label).reshape(10,self.batch_size), dtype=torch.float)
        es_bp, dws_bp = self.backprop_infer(inp, label,loss_fn_str = loss_fn)
        es_fp, dws_fp = self.gradient_infer(inp, label, loss_fn_str = loss_fn)
        for i in range(len(es_bp)):
            print("BP:")
            print(es_bp[i].shape)
            print(es_bp[i])
            print("Reconf:")
            print(es_fp[i].shape)
            print(es_fp[i])
        for i in range(len(self.pc_layers)):
            param_example = self.pc_layers[0].dparams
            N = len(param_example)
            for n in range(N):
                print("BP: ", dws_bp[i][n].shape)
                print(dws_bp[i][n])
                print("Reconfiguration :", self.pc_layers[i].dparams[n].shape)
                print(self.pc_layers[i].dparams[n])
  
    def gradient_infer(self, inp, label,loss_fn, loss_fn_str = "mse", store_evolutions = False, lambda_weight=1):
        inp = inp.to(self.device)
        label = label.to(self.device)
        out = self.forward(inp)
        es = [[] for l in range(len(self.pc_layers)+1)]
        dparamss = [torch.zeros(1) for i in range(len(self.pc_layers))]
        
        loss, dl = taf.vjp(lambda out: loss_fn(out, label),out,torch.tensor(1).to(self.device))
        es[-1] = deepcopy(dl) * lambda_weight
        es[0] = torch.zeros(1)
        if store_evolutions:
            xss = [[] for i in range(len(self.pc_layers))]
            ess = []
        for i in range(self.N_reconf_steps):
            for l in reversed(range(len(self.pc_layers)-1)):
                es[l+1] = self.pc_layers[l+1].x - self.pc_layers[l].forward(self.pc_layers[l].x)
                #print(es[l+1])
                dx, dparams = self.pc_layers[l+1].backward(es[l+2])
                dparamss[l+1] = deepcopy(dparams)
                self.pc_layers[l+1].x -= self.mu_dt * (es[l+1] - dx)
                
            out = self.pc_layers[-1].forward(self.pc_layers[-1].x)
            loss, dl = taf.vjp(lambda out: loss_fn(out,label),deepcopy(out),torch.tensor(1).to(self.device))
            es[-1] = deepcopy(-dl) * lambda_weight
            
            dx, dparams = self.pc_layers[0].backward(es[1])
            dparamss[0] = deepcopy(dparams)
            xs = []
            if store_evolutions:
                for i,l in enumerate(self.pc_layers):
                    xss[i].append(deepcopy(l.x))
                ess.append(deepcopy(es))
        self.dws = dparamss
        if store_evolutions:
            return es, dparamss, xss, ess
        return es,dparamss

    def get_parameters(self):
        param_list = []
        for l in self.pc_layers:
            for p in l.params:
                param_list.append(p)
        return param_list

    def init_optimizer(self, opt):
        self.opt = opt
        param_list = self.get_parameters()
        self.opt.params = param_list

    def step(self):
        if self.opt is None:
            raise ValueError("Must provide an optimizer to step using the init_optimizer function")
        idx = 0
        for i,l in enumerate(self.pc_layers):
            for j,p in enumerate(l.params):
                self.opt.params[idx].grad = -self.dws[i][j]
                idx +=1
        self.opt.step()

    def update_params(self,lr=None, store_updates = False):
        if lr is None:
            lr = self.lr
        if store_updates:
            dws = []
            for i,l in enumerate(self.pc_layers):
                dw = l.update_params(lr,fake_update = True)
                dws.append(dw)
            return dws
        else:
            for i,l in enumerate(self.pc_layers):
                dw = l.update_params(lr)
                
    def compute_test_accuracy(self, testset):
        N = 0
        total_acc = 0
        for i, (images, labels) in enumerate(testset):
            images = images.view(images.shape[0], -1)
            #onehot_labels = self.onehot_batch(labels).permute(1,0).float()
            out = self.forward(images)
            acc = accuracy(out, labels)
            N += 1
            total_acc += acc
        return total_acc / N
        

    def train(self, trainset, testset = None, N_epochs = 10,direct_bp = True, loss_fn_str = "mse",print_outputs = True, compute_test_acc = False, save_activity_differences_inference = True):
        losses = []
        accs = []
        test_accs = []
        activity_diffs = []
        for n in range(N_epochs):
            print("EPOCH ", n)
            for i, (img, label) in enumerate(trainset):
                img = torch.tensor(img.reshape(784,self.batch_size),dtype=torch.float).to(self.device)
                if loss_fn_str == "mse":
                    onehotted_label = torch.tensor(self.onehot_batch(label).reshape(10,self.batch_size), dtype=torch.float).to(self.device)
                else:
                    onehotted_label = label.to(self.device)
                if save_activity_differences_inference:
                    # save initial activities
                    init_activations = []
                    out = self.forward(img)
                    for l in self.pc_layers:
                        init_activations.append(deepcopy(l.x))
                # inference
                if self.use_reconfiguration:
                    es,dparams = self.gradient_infer(img, onehotted_label,loss_fn_str = loss_fn_str)
                if self.use_fp_reconfiguration:
                    es,dparams = self.fp_infer(img,onehotted_label,loss_fn_str = loss_fn_str)
                if self.use_backprop:
                    es,dparams = self.backprop_infer(img, onehotted_label,loss_fn_str = loss_fn_str)
                    
                if save_activity_differences_inference:
                    post_activations = []
                    for l in self.pc_layers:
                        post_activations.append(deepcopy(l.x))
                    layer_diffs = []
                    for i in range(len(self.pc_layers)):
                        layer_diffs.append(torch.sum(torch.square(post_activations[i] - init_activations[i])))
                    activity_diffs.append(np.array(layer_diffs))
                #weight update
                if not direct_bp:
                    self.update_params()
                if direct_bp:
                    self.set_params()
                    out = self.forward(img)
                if loss_fn_str == "mse":
                    loss_fn = nn.MSELoss(reduction="sum")
                if loss_fn_str == "crossentropy":
                    loss_fn = nn.CrossEntropyLoss(reduction="sum")
                    out =  out.T
        
                loss = loss_fn(out, onehotted_label)
                loss.backward()
                # iterate over all param list
                for i,l in enumerate(self.pc_layers):
                    for j in range(len(l.params)):
                        dparam = l.params[j].grad
                        l.params[j] = l.params[j].detach()
                        l.params[j] = l.params[j] - (self.lr *dparam.detach())
                        l.params[j] = nn.Parameter(l.params[j])
                out = self.forward(img)
                acc = self.batch_accuracy(out, label)
                if print_outputs:
                    print("acc: ", acc)
                    accs.append(acc)
                if loss_fn_str == "mse":
                    loss_fn = nn.MSELoss(reduction="sum")
                if loss_fn_str == "crossentropy":
                    loss_fn = nn.CrossEntropyLoss(reduction="sum")
                out =  out.T
                #loss = torch.sum(torch.square(out - onehotted_label)).item()
                loss = loss_fn(out, onehotted_label).item()
                losses.append(loss)
                if print_outputs:
                    print("loss: ", loss)
                if compute_test_acc is True and testset is not None:
                    test_acc = self.compute_test_accuracy(testset)
                    test_accs.append(test_acc)

        return np.array(losses), np.array(accs), np.array(test_accs), np.array(activity_diffs)