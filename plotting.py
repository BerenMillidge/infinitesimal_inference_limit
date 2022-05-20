import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *
from pcn import *

sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)

import brewer2mpl

bmap = brewer2mpl.get_map("Set2", 'qualitative',7)
colors = bmap.mpl_colors

def smoothing_filter(x,window_size):
    return np.convolve(x, np.ones(window_size)/window_size, mode='valid')

def plot_distances():
    distances = np.load("data/distance_list.npy")
    xs = np.arange(0, len(distances[0,:]))
    mean_distances = np.mean(distances, axis=0)
    std_distances = np.std(distances, axis=0)
    total_std = np.std(distances)
    print(total_std)
    print(distances.shape)
    # some quick outlier removal
    mean_mean_dist = np.mean(mean_distances)
    filtered_mean_dists = []
    for mdist in mean_distances:
        if mdist > mean_mean_dist + (3 * total_std):
            mdist = mean_mean_dist + (3 * total_std)
        filtered_mean_dists.append(mdist)

    std_distances = np.clip(std_distances, -3 * total_std, 3 * total_std)
    plt.plot(xs,filtered_mean_dists)
    plt.fill_between(xs, filtered_mean_dists - std_distances, filtered_mean_dists + std_distances, alpha=0.5)
    plt.show()

def plot_cosine_similarities():
    fig = plt.figure(figsize=(12,10))
    cosine_similarities = np.load("data/cosine_sim_list.npy")
    mean_sims = np.abs(np.mean(cosine_similarities,axis=0))
    std_sims = np.std(cosine_similarities, axis=0) / np.sqrt(len(cosine_similarities))
    mean_sims = np.clip(mean_sims, 0.9985, 1)
    std_sims = np.clip(std_sims, -3 * np.std(std_sims), 3 * np.std(std_sims))
    plt.ylim([0.995,1])
    xs = np.arange(0, len(mean_sims))
    plt.plot(xs, mean_sims)
    plt.fill_between(xs, mean_sims - std_sims, mean_sims + std_sims, alpha=0.5,color = colors[2])
    plt.xlabel("Batch Number",fontsize=25)
    plt.ylabel("Similarity Score",fontsize=25)
    sns.despine(left=False,top=True, right=True, bottom=False)
    plt.title("Similarity to backprop gradients during training",fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    fig.tight_layout()
    plt.savefig("figures/cosine_sims_plot.jpg")
    plt.show()

def plot_losses():
    bp_losses = np.load("data/bp_loss_list.npy")
    pc_losses = np.load("data/pc_loss_list.npy")
    mean_bp_losses = smoothing_filter(np.mean(bp_losses, axis=0),1)
    std_bp_losses = smoothing_filter(np.std(bp_losses, axis=0) / np.sqrt(len(mean_bp_losses)),1)
    xs = np.arange(0, len(mean_bp_losses))
    mean_pc_losses = smoothing_filter(np.mean(pc_losses, axis=0),1)
    std_pc_losses = smoothing_filter(np.std(pc_losses, axis=0) / np.sqrt(len(mean_bp_losses)),1)
    plt.plot(xs, mean_bp_losses, label="Backprop",alpha=0.7)
    plt.fill_between(xs, mean_bp_losses - std_bp_losses, mean_bp_losses + std_bp_losses, alpha=0.5)
    plt.plot(xs, mean_pc_losses, label="PC-Nudge",alpha=0.7)
    plt.fill_between(xs, mean_pc_losses - std_pc_losses, mean_pc_losses + std_pc_losses, alpha=0.5)
    plt.legend()
    plt.show()

def plot_accs():
    fig = plt.figure(figsize=(12,10))
    bp_losses = np.load("data/bp_acc_list.npy")
    pc_losses = np.load("data/pc_acc_list.npy")
    mean_bp_losses = smoothing_filter(np.mean(bp_losses, axis=0),5)
    std_bp_losses = smoothing_filter(np.std(bp_losses, axis=0) / np.sqrt(len(mean_bp_losses)),5)
    xs = np.arange(0, len(mean_bp_losses))
    mean_pc_losses = smoothing_filter(np.mean(pc_losses, axis=0),5)
    std_pc_losses = smoothing_filter(np.std(pc_losses, axis=0) / np.sqrt(len(mean_bp_losses)),5)

    plt.plot(xs, mean_pc_losses, label="PC-Nudge",alpha=1, color = colors[1])
    plt.fill_between(xs, mean_pc_losses - std_pc_losses, mean_pc_losses + std_pc_losses, alpha=0.5,color=colors[1])
    plt.plot(xs, mean_bp_losses, label="Backprop",alpha=0.5, color=colors[2])
    plt.fill_between(xs, mean_bp_losses - std_bp_losses, mean_bp_losses + std_bp_losses, alpha=0.5,color=colors[2])
    plt.legend(fontsize=25)
    plt.xlabel("Batch Number",fontsize=25)
    plt.ylabel("Similarity Score",fontsize=25)
    sns.despine(left=False,top=True, right=True, bottom=False)
    plt.title("Similarity to backprop gradients during training",fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    fig.tight_layout()
    plt.savefig("figures/mnist_acc_plot.jpg")
    plt.show()


def plot_lambda_backprop_distances(N_runs,lambda_weights,trainloader, input_size, hidden_sizes, output_size, batch_size, save_figure=True): 
    distance_mat = []
    for i in range(N_runs):
        distances = compute_distances(lambda_weights,trainloader, input_size, hidden_sizes, output_size, batch_size)
        distance_mat.append(np.array(distances))
    distance_mat = np.array(distance_mat)
    distances_mean = np.mean(distance_mat, axis=0)
    distances_std = np.std(distance_mat, axis=0) / np.sqrt(N_runs)
    print(distances)
    fig = plt.figure(figsize=(12,10))
    sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    plt.plot(lambda_weights, distances_mean)
    plt.fill_between(lambda_weights, distances_mean - distances_std, distances_mean + distances_std, alpha=0.5)
    plt.xlabel("Weighting Coefficient",fontsize=25)
    plt.ylabel("Total Normalized Euclidean Distance",fontsize=25)
    sns.despine(left=False,top=True, right=True, bottom=False)
    plt.title("Distance from true backprop gradient by lambda",fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    fig.tight_layout()
    if save_figure:
        plt.savefig("lambda_weighting_coeff_fig.jpg")
    plt.show()
    return distances_mean, distances_std, distance_mat


def plot_lambda_activity_equilibrium_distances(N_runs, lambda_weights,trainloader, input_size, hidden_sizes, output_size, batch_size, save_figure = True):
    distance_mat = []
    for i in range(N_runs):
        distances = compute_equilibrium_activity_differences(lambda_weights,trainloader, input_size, hidden_sizes, output_size, batch_size)
        distance_mat.append(np.array(distances))
    distance_mat = np.array(distance_mat)
    print(distance_mat.shape)
    distances_mean = np.mean(distance_mat, axis=0)
    distances_std = np.std(distance_mat, axis=0) / np.sqrt(N_runs)
    # this is just a bunch of straightforward graphing
    # Im very sure that `nudge' PC will work. and we have discovered an approximately linear relationship here
    print(distances)
    fig = plt.figure(figsize=(12,10))
    sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    plt.plot(lambda_weights, distances_mean)
    plt.fill_between(lambda_weights, distances_mean - distances_std, distances_mean + distances_std, alpha=0.5)
    plt.xlabel("Weighting Coefficient",fontsize=25)
    plt.ylabel("Total Euclidean Distance",fontsize=25)
    sns.despine(left=False,top=True, right=True, bottom=False)
    plt.title("Distance from Free Phase Equilibrium by lambda",fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    fig.tight_layout()
    if save_figure:
        plt.savefig("activity_equilibrium_distances_fig.jpg")
    plt.show()
    return distances_mean, distances_std, distance_mat


def make_distance_distance_plot(eq_mat, grad_mat):
    fig = plt.figure(figsize=(12,10))
    for i in range(10):
        plt.plot(eq_mat[i,:], grad_mat[i,:], label="Initialization " + str(i))
    #plt.legend(fontsize=22)
    plt.xlabel("Distance from Free Phase",fontsize=25)
    plt.ylabel("Distance from Backprop Gradient",fontsize=25)
    sns.despine(left=False,top=True, right=True, bottom=False)
    plt.title("Distance from Backprop by distance from equilibrium",fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    fig.tight_layout()
    plt.savefig("distance_distance_plot.jpg")
    plt.show()

def plot_backprop_inference_distance_graph(N_runs,trainloader, input_size, hidden_sizes,output_size, batch_size):
    distance_mat = []
    for i in range(N_runs):
        distances = compute_inference_backprop_distances(trainloader, input_size, hidden_sizes,output_size, batch_size)
        distances = np.array(distances)
        distance_mat.append(deepcopy(distances))
    distance_mat = np.array(distance_mat)
    mean_distances = np.mean(distance_mat, axis=0)
    std_distances = np.std(distance_mat, axis=0) / np.sqrt(N_runs)
    fig = plt.figure(figsize=(12,10))
    xs = np.arange(0,T)
    plt.plot(xs,mean_distances)
    plt.fill_between(xs, mean_distances - std_distances, mean_distances + std_distances, alpha=0.5)
    plt.xlabel("Inference Timestep",fontsize=25)
    plt.ylabel("Euclidean Distance to Backprop Gradients",fontsize=25)
    plt.title("Distance from Backprop Gradients During Inference", fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.legend(fontsize=28)
    fig.tight_layout()
    plt.savefig("backprop_distances_during_inference.jpg")
    plt.show()
    
def plot_energies_evolution(Fs, out_Ls, E_tildes):
    fig = plt.figure(figsize=(12,10))
    xs = np.arange(len(Fs))
    sns.set_theme(context='talk',font='sans-serif',font_scale=1.0)
    plt.plot(xs,Fs, label="Total Energy")
    plt.plot(xs,out_Ls, label="Backprop Loss")
    plt.plot(xs,E_tildes, label="Internal Energy")
    plt.xlabel("Inference timesteps",fontsize=25)
    plt.ylabel("Energy Value",fontsize=25)
    sns.despine(left=False,top=True, right=True, bottom=False)
    plt.title("Evolution of energy components during inference",fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=28)
    fig.tight_layout()
    plt.savefig("energies_evolution_fig.jpg")
    plt.show()
    
    
    
if __name__ == '__main__':
    input_size = 784
    hidden_sizes = [128, 64]
    #hidden_sizes = [100,100]
    output_size = 10
    batch_size  = 64
    N_training_runs = 5
    N_plot_runs = 10
    lambda_weights = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    trainloader, valloader = load_mnist_data(batch_size)
    
    
    model, pcnet, images, labels = make_model_data(trainloader, input_size, hidden_sizes, output_size, batch_size)
    
    # inference energies plot
    Fs, out_Ls, E_tildes = get_energies_during_inference(pcnet, images, labels)
    plot_energies_evolution(Fs, out_Ls, E_tildes)
    
    # general distance plots
    eq_dists, eq_stds, eq_mat = plot_lambda_activity_equilibrium_distances(N_plot_runs, lambda_weights,trainloader, input_size, hidden_sizes, output_size, batch_size)
    grad_dists, grad_stds, grad_mat = plot_lambda_backprop_distances(N_plot_runs, lambda_weights,trainloader, input_size, hidden_sizes, output_size, batch_size)
    make_distance_distance_plot(eq_mat, grad_mat)
    #plot_backprop_inference_distance_graph(N_plot_runs,trainloader, input_size, hidden_sizes,output_size, batch_size)
    
    # run training experiments
    pc_loss_list, pc_acc_list, bp_loss_list, bp_acc_list, cosine_sim_list, distance_list = run_training_experiment(N_training_runs,trainloader, input_size, hidden_sizes, output_size, batch_size)
    plot_cosine_similarities()
    plot_accs()
    plot_losses()
    plot_distances()