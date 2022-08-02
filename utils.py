# general utility functions loading PCNs integrating with PCNs etc


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
import os
from pcn import *


def load_mnist_data(batch_size):
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

    if not os.path.exists("data/"):
        os.makedirs("data/")
    trainset = datasets.MNIST('data/mnist_train', download=True, train=True, transform=transform)
    valset = datasets.MNIST('data/mnist_test', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
    return trainloader, valloader


def mse_loss(x,y):
    return 0.5 * torch.sum(torch.sum(torch.square(x-y),dim=[1]),dim=[0])

def get_pc_bp_params(images, labels,model,pcnet,lambda_weighting = 1, return_xs=False):
    criterion = mse_loss

    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.0)
    opt_pc = optim.SGD(pcnet.get_parameters(), lr=0.005, momentum=0.0)
    pcnet.init_optimizer(opt_pc)
    bp_params = []
    pc_params = []
    if return_xs:
        bp_xs = []
        pc_xs = []
    es, dws_bp = pcnet.backprop_infer(images, labels,loss_fn = criterion)
    for dw_bp in list(dws_bp):
        for d_bp in dw_bp:
            bp_params.append(deepcopy(d_bp))
    if return_xs:
        for l in pcnet.pc_layers:
            bp_xs.append(deepcopy(l.x))
    
    es, dws = pcnet.gradient_infer(images, labels, loss_fn = criterion,lambda_weight=lambda_weighting)
    for dw in list(dws):
        for d in dw:
            #print(d.shape)
            #print(d[0:10])
            pc_params.append(deepcopy(-d))
    if return_xs:
        for l in pcnet.pc_layers:
            pc_xs.append(deepcopy(l.x))
    if return_xs:
        return bp_params, pc_params, bp_xs, pc_xs
    else:
        return bp_params, pc_params




def compute_inference_backprop_distances(trainloader, input_size, hidden_sizes,output_size, batch_size):
    model = NN_model(input_size, hidden_sizes, output_size, batch_size)
    w1, b1 = list(model.fc1.parameters())
    w2, b2 = list(model.fc2.parameters())
    w3,b3 = list(model.fc3.parameters())
    pc_fc1 = PCLayer2_GPU("cpu", linear_base_function,[w1.detach(),b1.detach()],f=torch.relu)
    pc_fc2 = PCLayer2_GPU("cpu", linear_base_function,  [w2.detach(),b2.detach()],f=torch.relu)
    #pc_fc3 = PCLayer2_GPU("cpu", linear_base_function, [w3.detach(),b3.detach()],f=nn.LogSoftmax(dim=1))
    pc_fc3 = PCLayer2_GPU("cpu", linear_base_function, [w3.detach(),b3.detach()],f=nn.Identity())
    pcnet = PC_Net2_GPU([pc_fc1, pc_fc2, pc_fc3],batch_size, mu_dt = 0.1, lr = 0.1,fp_lr = 1,N_fp_steps = 1,N_reconf_steps = 100,use_backprop=False, clamp_val=10000)
    images, labels = next(iter(trainloader))
    images = images.view(images.shape[0], -1)[0:1,:]
    print("LABELS PRE: ", labels.shape)
    labels = pcnet.onehot_batch(labels).permute(1,0).float()[0:1,:]
    print("LABELS POST: ", labels.shape)
    images2 = deepcopy(images)
    logps = model(images) #log probabilities
    print("LOGPS: ", logps.shape)
    logps_pc = pcnet.forward(images2)
    print("BP NET: ", logps[0,:])
    print("PC NET: ", logps_pc[0,:])
    #criterion = nn.NLLLoss()
    #criterion = nn.MSELoss()
    criterion = mse_loss

    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.0)
    opt_pc = optim.SGD(pcnet.get_parameters(), lr=0.005, momentum=0.0)
    pcnet.init_optimizer(opt_pc)
    optimizer.zero_grad()
    loss = criterion(logps, labels)
    print("loss 1",loss.item())
    loss.backward()
    for i,p in enumerate(model.parameters()):
        print("Param: ", i, p.grad.shape)
        print(p.grad[0:10])

    print("     PC GRADS    ")
    #es, dws = pcnet.backprop_infer(images, labels,loss_fn = criterion)
    #es, dws = pcnet.fp_infer_3(images, labels, loss_fn = criterion)
    es, dws = pcnet.gradient_infer(images, labels, loss_fn = criterion)
    for dw in list(dws):
        for d in dw:
            print(d.shape)
            print(d[0:10])

    es, dws, xss, ess = pcnet.gradient_infer(images, labels, loss_fn = criterion, store_evolutions = True)
    
    #for i in range(len(ess)):
    #    print("ES ",i)
    #    for j in range(len(ess[i])):
    #        print(ess[i][j].shape)
        

    #print(len(ess))
    #print(len(ess[0]))
    #print(len(ess[0][0]))
    #print(len(ess[1]))
    #print(len(ess[1][0]))
    #print(ess[0].shape)
    #print(ess[0,:])
    #ess = np.array(ess)
    #print(ess[1][1])

    T = len(ess)
    L = len(ess[0])
    distances = []
    for t in range(T):
        total_dist = 0
        for l in range(L):
            total_dist += torch.sum(torch.square(ess[t][l] - ess[0][l])).item()
        distances.append(deepcopy(total_dist))
    return distances


def make_model_data(trainloader,input_size, hidden_sizes, output_size, batch_size):
    model = NN_model(input_size, hidden_sizes, output_size, batch_size)
    print(model)

    w1, b1 = list(model.fc1.parameters())

    w2, b2 = list(model.fc2.parameters())
    w3,b3 = list(model.fc3.parameters())
    pc_fc1 = PCLayer2_GPU("cpu", linear_base_function,[w1.detach(),b1.detach()],f=torch.relu)
    pc_fc2 = PCLayer2_GPU("cpu", linear_base_function,  [w2.detach(),b2.detach()],f=torch.relu)
    #pc_fc3 = PCLayer2_GPU("cpu", linear_base_function, [w3.detach(),b3.detach()],f=nn.LogSoftmax(dim=1))
    pc_fc3 = PCLayer2_GPU("cpu", linear_base_function, [w3.detach(),b3.detach()],f=nn.Identity())
    pcnet = PC_Net2_GPU([pc_fc1, pc_fc2, pc_fc3],batch_size, mu_dt = 0.1, lr = 0.1,fp_lr = 1,N_fp_steps = 1,N_reconf_steps = 50,use_backprop=False, clamp_val=10000)

    images, labels = next(iter(trainloader))
    images = images.view(images.shape[0], -1)[0:1,:]
    print("LABELS PRE: ", labels.shape)
    labels = pcnet.onehot_batch(labels).permute(1,0).float()[0:1,:]
    return model, pcnet, images, labels


def get_energies_during_inference(pcnet, images, labels,criterion = mse_loss):
    print(images.shape)
    print(labels.shape)
    es, dws, xss, ess = pcnet.gradient_infer(images, labels, loss_fn = criterion, store_evolutions = True, lambda_weight=1)
    print(len(xss))
    print(len(ess[0]))
    Fs = []
    out_Ls = []
    E_tildes = []
    for t in range(len(ess)):
        es = ess[t]
        E_tilde = torch.sum(torch.square(es[1])) + torch.sum(torch.square(es[2]))
        out_l = torch.sum(torch.square(es[3])).item()
        Fe = E_tilde.item() + out_l
        Fs.append(Fe)
        out_Ls.append(out_l)
        E_tildes.append(E_tilde.item())
    return Fs, out_Ls, E_tildes



def compute_equilibrium_activity_differences(lambda_weights, trainloader, input_size, hidden_sizes, output_size, batch_size):
    model = NN_model(input_size, hidden_sizes, output_size, batch_size)
    ex_img = torch.randn(784).reshape(1,28,28)
    w1, b1 = list(model.fc1.parameters())
    w2, b2 = list(model.fc2.parameters())
    w3,b3 = list(model.fc3.parameters())
    pc_fc1 = PCLayer2_GPU("cpu", linear_base_function,[w1.detach(),b1.detach()],f=torch.relu)
    pc_fc2 = PCLayer2_GPU("cpu", linear_base_function,  [w2.detach(),b2.detach()],f=torch.relu)
    #pc_fc3 = PCLayer2_GPU("cpu", linear_base_function, [w3.detach(),b3.detach()],f=nn.LogSoftmax(dim=1))
    pc_fc3 = PCLayer2_GPU("cpu", linear_base_function, [w3.detach(),b3.detach()],f=nn.Identity())
    pcnet = PC_Net2_GPU([pc_fc1, pc_fc2, pc_fc3],batch_size, mu_dt = 0.1, lr = 0.001,fp_lr = 1,N_fp_steps = 1,N_reconf_steps = 50,use_backprop=False, clamp_val=10000)
    images, labels = next(iter(trainloader))
    images = images.view(images.shape[0], -1)[0:1,:]
    labels = pcnet.onehot_batch(labels).permute(1,0).float()[0:1,:]
    images2 = deepcopy(images)
    logps = model(images) 
    logps_pc = pcnet.forward(images2)

    distances = []
    for lambda_weight in lambda_weights:
        bp_params, pc_params, bp_xs, pc_xs = get_pc_bp_params(images, labels,deepcopy(model), deepcopy(pcnet),lambda_weighting=lambda_weight, return_xs = True)
        total_dist = 0
        for i,(bp_p, pc_p) in enumerate(zip(bp_xs, pc_xs)):
            print("layer " + str(i) + " " + str(torch.sum(torch.square(((bp_p) - (pc_p))))))
            total_dist += torch.sum(torch.square(((bp_p) - (pc_p))))
        distances.append(total_dist.item())
    return distances

def compute_distances(lambda_weights, trainloader, input_size, hidden_sizes, output_size, batch_size):
    model = NN_model(input_size, hidden_sizes, output_size, batch_size)
    ex_img = torch.randn(784).reshape(1,28,28)
    w1, b1 = list(model.fc1.parameters())
    w2, b2 = list(model.fc2.parameters())
    w3,b3 = list(model.fc3.parameters())
    pc_fc1 = PCLayer2_GPU("cpu", linear_base_function,[w1.detach(),b1.detach()],f=torch.relu)
    pc_fc2 = PCLayer2_GPU("cpu", linear_base_function,  [w2.detach(),b2.detach()],f=torch.relu)
    #pc_fc3 = PCLayer2_GPU("cpu", linear_base_function, [w3.detach(),b3.detach()],f=nn.LogSoftmax(dim=1))
    pc_fc3 = PCLayer2_GPU("cpu", linear_base_function, [w3.detach(),b3.detach()],f=nn.Identity())
    pcnet = PC_Net2_GPU([pc_fc1, pc_fc2, pc_fc3],batch_size, mu_dt = 0.1, lr = 0.001,fp_lr = 1,N_fp_steps = 1,N_reconf_steps = 50,use_backprop=False, clamp_val=10000)
    images, labels = next(iter(trainloader))
    images = images.view(images.shape[0], -1)[0:1,:]
    labels = pcnet.onehot_batch(labels).permute(1,0).float()[0:1,:]
    images2 = deepcopy(images)
    logps = model(images)
    logps_pc = pcnet.forward(images2)

    distances = []
    for lambda_weight in lambda_weights:
        bp_params, pc_params = get_pc_bp_params(images, labels,deepcopy(model), deepcopy(pcnet),lambda_weighting=lambda_weight)
        print("PC: ", pc_params[-1])
        print("BP: ", bp_params[-1] * lambda_weight)
        total_dist = 0
        for bp_p, pc_p in zip(bp_params, pc_params):
            total_dist += torch.sum(torch.square(((bp_p) - (pc_p/lambda_weight)))) # normalize by lambda weight 
        distances.append(total_dist.item())
    return distances


### run the learning experiment -- training the network on mnist

def accuracy(out, labels):
    with torch.no_grad():
        maxes = torch.argmax(out.detach(), dim=1)
        corrects = maxes == labels
        return torch.sum(corrects).item() / len(corrects) 

def batched_cosine_similarity(x,y,cosine=False):
    # assume batch is first dimension
    similarities = []
    assert x.shape == y.shape, "must have same shape"
    for i in range(len(x)):
        sim = torch.dot(x[i,:],y[i,:]) / ((torch.norm(x[i,:]) + eps) * torch.norm(y[i,:])+eps)
        if cosine:
        #print(sim)
            sim = torch.acos(sim)
            similarities.append(sim.detach().cpu().numpy())
    similarities = np.array(similarities)
    mean_similarities = np.mean(similarities)
    std_similarities = np.std(similarities)
    return mean_similarities, std_similarities

def cosine_similarity(x,y,cosine=False):
    x = x.flatten()
    y = y.flatten()
    eps = 1e-6
    sim = torch.dot(x,y) / ((torch.norm(x) + eps) * (torch.norm(y) + eps))
    #print(sim.item())
    return sim.item()

def train_network(trainloader, testloader, input_size, hidden_sizes, output_size, batch_size,N_epochs = 5, use_test_accuracy=False, save_activity_differences_inference = False):
    model = NN_model(input_size, hidden_sizes, output_size, batch_size)
    w1, b1 = list(model.fc1.parameters())
    w2, b2 = list(model.fc2.parameters())
    w3,b3 = list(model.fc3.parameters())

    pc_fc1 = PCLayer2_GPU("cpu", linear_base_function,[deepcopy(w1.detach()),deepcopy(b1.detach())],f=torch.relu)
    pc_fc2 = PCLayer2_GPU("cpu", linear_base_function,  [deepcopy(w2.detach()),deepcopy(b2.detach())],f=torch.relu)
    #pc_fc3 = PCLayer2_GPU("cpu", linear_base_function, [deepcopy(w3.detach()),deepcopy(b3.detach())],f=nn.LogSoftmax(dim=1))
    pc_fc3 = PCLayer2_GPU("cpu", linear_base_function, [deepcopy(w3.detach()),deepcopy(b3.detach())],f=nn.Identity())
    pcnet = PC_Net2_GPU([pc_fc1, pc_fc2, pc_fc3],batch_size, mu_dt = 0.3,fp_lr = 0.2, lr = 0.0001,N_fp_steps = 20,N_reconf_steps = 50,use_backprop=False, clamp_val=10000)
    #criterion = nn.NLLLoss()
    criterion = mse_loss
    #criterion = nn.MSELoss()

    PLOT_DYNAMICS = False
    LAMBDA_VALUE = 0.0001

    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    optimizer_pc = optim.SGD(pcnet.get_parameters(), lr=0.0001 / LAMBDA_VALUE, momentum=0.9)
    pcnet.init_optimizer(optimizer_pc)
    optimizer.zero_grad()
    time0 = time()
    accs = []
    cosine_similarities = []
    all_distances = [] 
    bp_accs = []
    pc_losses = []
    bp_losses = []
    test_accs_pc = []
    test_accs_bp = []
    activity_diffs = []
    for e in range(N_epochs):
        running_loss = 0
        running_loss_bp = 0
        for i,(images, labels) in enumerate(trainloader):
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            onehot_labels = pcnet.onehot_batch(labels).permute(1,0).float()
        
            # Training pass
            optimizer.zero_grad()
                    #with torch.no_grad():
            output = pcnet.forward(images)
            loss = criterion(output, onehot_labels)
            #print("loss: ", loss.item())
            pc_losses.append(loss.item())
            acc = accuracy(output, labels)
            accs.append(acc)
            print("PC acc: ", acc)
            # save activity of inference net beforehand
            if save_activity_differences_inference:
                # save initial activities
                init_activations = []
                out = pcnet.forward(images)
                for l in pcnet.pc_layers:
                    init_activations.append(deepcopy(l.x))

            es, dws_bp =pcnet.backprop_infer(images, onehot_labels,loss_fn = criterion)

            es, dws = pcnet.gradient_infer(images, onehot_labels, loss_fn = criterion,lambda_weight = LAMBDA_VALUE)
            
            # save activity of inference net after inference is complete
            if save_activity_differences_inference:
                post_activations = []
                for l in pcnet.pc_layers:
                    post_activations.append(deepcopy(l.x))
                layer_diffs = []
                for i in range(len(pcnet.pc_layers)):
                    layer_diffs.append(torch.sum(torch.square(post_activations[i] - init_activations[i])))
                activity_diffs.append(np.array(layer_diffs))
                print("total activity diff: ", np.sum(np.square(np.array(layer_diffs))))
            cosine_sims = []
            distances = []
            for dw_pc, dw_bp in zip(dws,dws_bp):
                sim = cosine_similarity(dw_pc[0], dw_bp[0])
                distance = torch.sum(torch.square((dw_pc[0]/LAMBDA_VALUE) + dw_bp[0])).item()
                #print("PC: ", dw_pc[0] / LAMBDA_VALUE)
                #print("BP: ", dw_bp[0])
                #print("Distance: ",distance)
                cosine_sims.append(sim)
                distances.append(distance)
            mean_cosine_sim = np.mean(np.array(cosine_sims))
            #print("Cosine Similarities: ", np.mean(np.array(cosine_sims)))
            cosine_similarities.append(mean_cosine_sim)
            all_distances.append(np.mean(np.array(distances)))

            pcnet.step()

            output_bp = model(images)
            loss_bp = criterion(output_bp, onehot_labels)
            bp_losses.append(loss_bp.item())
            #print("LOSS BP: ", loss_bp.item())
            acc_bp = accuracy(output_bp, labels)
            bp_accs.append(acc_bp)
            print("BP acc: ", acc_bp)
            loss_bp.backward()
            optimizer.step()

            running_loss += loss.item()
            running_loss_bp += loss_bp.item()
            if use_test_accuracy is True and testloader is not None:
                test_acc_pc = pcnet.compute_test_accuracy(testloader)
                test_accs_pc.append(test_acc_pc)
                print("PC test acc: ", test_acc_pc)
                
                test_acc_bp = model.compute_test_accuracy(testloader)
                test_accs_bp.append(test_acc_bp)
                print("BP test acc:", test_acc_bp)
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
            #print("RUNNING LOSS BP: ", running_loss_bp)
            print("\nTraining Time (in minutes) =",(time()-time0)/60)
            print("acc: ", acc)
            print("acc bp: ", acc_bp)
    return pc_losses, accs, bp_losses, bp_accs, cosine_similarities, all_distances, test_accs_pc, test_accs_bp, activity_diffs


def run_training_experiment(N_runs, trainloader, testloader, input_size, hidden_sizes, output_size, batch_size,N_epochs = 5,sname = "run_3",use_test_accuracy=False, save_activity_differences_inference = False):
    pc_loss_list = []
    pc_acc_list = []
    bp_loss_list = []
    bp_acc_list = []
    cosine_sim_list = []
    distance_list = []
    test_pc_acc_list = []
    test_bp_acc_list = []
    activity_diffs_list = []
    for i in range(N_runs):
        pc_losses, pc_accs, bp_losses, bp_accs, cosine_similarities, all_distances, test_accs_pc, test_accs_bp, activity_diffs = train_network(trainloader, testloader, input_size, hidden_sizes, output_size, batch_size,N_epochs = N_epochs,use_test_accuracy=use_test_accuracy, save_activity_differences_inference=save_activity_differences_inference)
        pc_loss_list.append(np.array(pc_losses))
        pc_acc_list.append(np.array(pc_accs))
        bp_loss_list.append(np.array(bp_losses))
        bp_acc_list.append(np.array(bp_accs))
        cosine_sim_list.append(np.array(cosine_similarities))
        distance_list.append(np.array(all_distances))
        test_pc_acc_list.append(np.array(test_accs_pc))
        test_bp_acc_list.append(np.array(test_bp_acc_list))
        activity_diffs_list.append(np.array(activity_diffs))
    
    pc_loss_list = np.array(pc_loss_list)
    pc_acc_list = np.array(pc_acc_list)
    bp_loss_list = np.array(bp_loss_list)
    bp_acc_list = np.array(bp_acc_list)
    cosine_sim_list = np.array(cosine_sim_list)
    distance_list = np.array(distance_list)
    test_pc_acc_list = np.array(test_pc_acc_list)
    test_bp_acc_list = np.array(test_bp_acc_list)
    activity_diffs_list = np.array(activity_diffs_list)
    if not os.path.exists("data/"):
        os.makedirs("data/")
    np.save("data/pc_loss_list_" + sname + ".npy", pc_loss_list)
    np.save("data/pc_acc_list_" + sname + ".npy", pc_acc_list)
    np.save("data/bp_loss_list_" + sname + ".npy", bp_loss_list)
    np.save("data/bp_acc_list_" + sname + ".npy", bp_acc_list)
    np.save("data/cosine_sim_list_" + sname + ".npy", cosine_sim_list)
    np.save("data/distance_list_" + sname + ".npy", distance_list)
    np.save("data/test_pc_acc_list_" + sname + ".npy", test_pc_acc_list)
    np.save("data/test_bp_acc_list_" + sname + ".npy", test_bp_acc_list)
    np.save("data/activity_diffs_list_" + sname + ".npy", activity_diffs_list)
    return pc_loss_list, pc_acc_list, bp_loss_list, bp_acc_list, cosine_sim_list, distance_list, test_pc_acc_list, test_bp_acc_list, activity_diffs_list


if __name__ == '__main__':
    input_size = 784
    hidden_sizes = [128, 64]
    #hidden_sizes = [100,100]
    output_size = 10
    batch_size  = 64
    N_training_runs = 5 # 5
    N_plot_runs = 10
    USE_TEST_ACCURACY = True
    SAVE_ACTIVITY_DIFFERENCES = True
    N_epochs = 1
    
    
    trainloader, valloader = load_mnist_data(batch_size)
    run_training_experiment(N_training_runs, trainloader, valloader,input_size, hidden_sizes, output_size,batch_size, N_epochs = N_epochs,use_test_accuracy=USE_TEST_ACCURACY,save_activity_differences_inference=SAVE_ACTIVITY_DIFFERENCES)