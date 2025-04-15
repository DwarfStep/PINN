## let's import the relevant libraries
import torch
import torch.nn as nn
from time import perf_counter
from PIL import Image
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import requests
import os
# данные для задачи

"""parametrs"""
nu = 0.3 # коэфициент пуасонна
a = 0.25 # внутренний радиус
b = 0.45 # внешний радиус
pi = -1 # внутреннее давление
p0 = -2 # внешнее давление
E = 2e11 # модуль Юнга
lam = nu*E/((1+nu)*(1-2*nu)) # коэфиициент Лямэ
mu = E/2*(1+nu) # коэфиициент Лямэ
device = torch.device("cpu")
a = 0.25
b = 0.45
phi_0 = 0
phi_1 = np.pi/2
num_1 = 20
num_2 = 40
size = 6

""" functions"""
def grad(outputs, inputs):
    '''функция которая берет производная'''
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]


def sig(x, x_i, x_0, pi, p0, network1, network2, scale_network_out):
    ''' 2d область для перемещений'''
    coef_1 = (p0-pi)/(x_0-x_i)
    coef_2 = -((p0-pi)/(x_0-x_i)*x_0) + p0
    sig_rho = coef_2 + x[:, 0][:,None]*coef_1 + (x[:, 0][:, None] - x_i)*(x[:, 0][:, None] - x_0)*network1(x)/scale_network_out
    sig_phi = network2(x)/scale_network_out
    sig = torch.cat((sig_rho, sig_phi), dim=1)
    return sig


def loss(X, sig):

    X.requires_grad = True
    sig = sig(X)
    g_sig_rho = grad(sig[:, 0], X)
    g_sig_rho_2 = grad(g_sig_rho[:, 0], X)
    loss = g_sig_rho_2[:, 0] + g_sig_rho[:, 0]/X[:,0] - sig[:, 0]/X[:,0]**2
    return torch.mean(loss)


def train_MultiFieldFracture_seperate_net(net1, net2, batch_size_X, max_iter,
                                          print_results_every, scale_network_out, x_0, x_i, num_y):

    print('\n\nStarting training loop with seperate networks for sig_rho and sig_phi...\n\n')
    print( '\t\tbatch_size_X: %d' % batch_size_X,
          '\t\tmax_iter: %d\n' % max_iter)
    network1 = net1
    network1 = network1.to(device)
    network2 = net2
    network2 = network2.to(device)

    parameters = list(network1.parameters()) + list(network2.parameters()) #the parameters to optimze
    optimizer = torch.optim.Adam(parameters, lr=1e-5)  # lr is the learning rate

    # Records time the loop starts
    start_time = perf_counter()
    loss_list = []
    elapsed_time = 0.0
    running_loss = 0.0
    #max_iter= 125*num_y
    for i in range(max_iter):
        X1 = torch.distributions.Uniform(x_i, x_0).sample((batch_size_X, 1))
        #X1 = torch.linspace(x_i, x_0, batch_size_X)[:,None]
        # X2 = torch.tensor(batch_size_X * [(i % 125) * np.pi / (2 * num_y)])[:, None]
        X2 = torch.distributions.Uniform(0, np.pi/2).sample((batch_size_X, 1))
        X = torch.cat((X1, X2), dim=1)
        X = X.to(device)
        optimizer.zero_grad()
        l = loss(X,
                 partial(sig, x_i = a, x_0 = b, pi = pi, p0 = p0, network1=network1, network2=network2, scale_network_out=scale_network_out))
        running_loss += l.item()
        l.backward()
        optimizer.step()
        if (i + 1) % print_results_every == 0:
            # Print loss, time elapsed every "print_results_every"# iterations
            current_time = perf_counter()
            elapsed_time = current_time - start_time
            print('[iter: %d]' % (i + 1), '\t\telapsed_time: %3d secs' % elapsed_time, '\t\tLoss: ',
                  running_loss / print_results_every)
            loss_list.append(running_loss / print_results_every)
            running_loss = 0.0

    return loss_list, network1, network2

def model_capacity(net):
    """
    Prints the number of parameters and the number of layers in the network
    -> Requires a neural network as input
    """
    number_of_learnable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    num_layers = len(list(net.parameters()))
    print("\nThe number of layers in the model: %d" % num_layers)
    print("The number of learnable parameters in the model: %d\n" % number_of_learnable_params)


# def plot_loss(loss, label):
#     """
#     Plots the loss function.
#     -> loss: list containing the losses
#     -> label: label for this loss
#     """
#     ax.plot(100 * np.arange(len(loss)), loss, label='%s' % label)
#     ax.set_xlabel('Iterations')
#     ax.set_ylabel('Loss')
#     plt.legend(loc='best')


def plot_displacement(network1, network2, scale_network_out, num_r, num_phi, s=20):
    """
    Plots the horizontal and vertical components of displacements at given number of points.
    -> network1: neural network 1
    -> network2: neural network 2
    -> num_x: number of grid points to r coordinate
    -> num_x: number of grid points to phi coordinate
    """

    X_init = torch.linspace(0.25, 0.45, num_r)
    Y_init = torch.linspace(0.0, np.pi/2, num_phi)
    X_init, Y_init = torch.meshgrid(X_init, Y_init)
    X_init = X_init.reshape(-1)[:, None]
    Y_init = Y_init.reshape(-1)[:, None]
    X_gl = torch.cat((X_init, Y_init), dim=1)
    press = sig(X_gl, a, b, pi, p0, network1, network2, scale_network_out).detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    c1 = ax.scatter(X_gl[:, 1], X_gl[:, 0], s=s, c=press[:, 0],
                        cmap='jet')
    plt.colorbar(c1, ax=ax)
    ax.set_title('Radian stress ($\sigma_rho$)')
    ax.set_thetamin(0)
    ax.set_thetamax(90)
    ax.set_rmax(0.46)
    plt.savefig("str_r.jpg")
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection='polar')
    c2 = ax1.scatter(X_gl[:, 1], X_gl[:, 0], s=s, c=press[:, 1],
                        cmap='jet')
    plt.colorbar(c2, ax=ax1)
    ax1.set_title('Radian stress ($\sigma_\phi$)')
    ax1.set_thetamin(0)
    ax1.set_thetamax(90)
    ax1.set_rmax(0.46)
    plt.savefig("str_shr.jpg")
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.plot(X_gl[0:-1:num_r, 0], press[0:-1:num_r, 0])
    plt.savefig("sig_r.jpg")
    """
    Scale down the network output to ensure we get +ve determinant of the Jacobian.
    We have to scale the output so that as the training begins we don't initialize a displacement
    which has no physical meaning.0
    For example, the determinant of the Jacobian cannot be negative
    since that would mean negative volume; which has no physical meaning.
    """

scale_network_out = 0.1


# here is the network for the sig_rho
simple_net1 = nn.Sequential(nn.Linear(2,50),
                     nn.Sigmoid(),
                     nn.Linear(50,50),
                     nn.Sigmoid(),
                     nn.Linear(50,50),
                     nn.Sigmoid(),
                     nn.Linear(50,1)
                    )

# here is the network for the sig_phi
simple_net2 = nn.Sequential(nn.Linear(2,50),
                     nn.Sigmoid(),
                     nn.Linear(50,50),
                     nn.Sigmoid(),
                     nn.Linear(50, 50),
                     nn.Sigmoid(),
                     nn.Linear(50,1)
                    )

# here is how we can find the number of layers and number of model parameters in each network
model_capacity(simple_net1)
model_capacity(simple_net2)

loss_list_simple_net, simple_net1, simple_net2 = train_MultiFieldFracture_seperate_net(net1=simple_net1, net2=simple_net2,
                                                                      batch_size_X = num_1,
                                                                      max_iter = 5000,
                                                                      print_results_every = 100, scale_network_out=scale_network_out, x_0= b, x_i = a, num_y=num_2)
# Let's visualize the training loss
# figure, ax = plt.subplots(dpi=100)
# plot_loss(loss_list_simple_net, 'Simple Net')
# ax.plot(100*np.arange(len(loss_list_simple_net)), 0.006*np.ones(len(loss_list_simple_net)), label='FEM Energy')

plot_displacement(simple_net1, simple_net2, scale_network_out,
                  num_1, num_2)