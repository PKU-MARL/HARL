"""Tools for HARL."""
import copy
import math
import torch
import torch.nn as nn


def init_device(args):
    """Init device.
    Args:
        args: (dict) arguments
    Returns:
        device: (torch.device) device
    """
    if args["cuda"] and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        if args["cuda_deterministic"]:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
    torch.set_num_threads(args["torch_threads"])
    return device


def get_active_func(activation_func):
    """Get the activation function.
    Args:
        activation_func: (str) activation function
    Returns:
        activation function: (torch.nn) activation function
    """
    if activation_func == "sigmoid":
        return nn.Sigmoid()
    elif activation_func == "tanh":
        return nn.Tanh()
    elif activation_func == "relu":
        return nn.ReLU()
    elif activation_func == "leaky_relu":
        return nn.LeakyReLU()
    elif activation_func == "selu":
        return nn.SELU()
    elif activation_func == "hardswish":
        return nn.Hardswish()
    elif activation_func == "identity":
        return nn.Identity()
    else:
        assert False, "activation function not supported!"


def get_init_method(initialization_method):
    """Get the initialization method.
    Args:
        initialization_method: (str) initialization method
    Returns:
        initialization method: (torch.nn) initialization method
    """
    return nn.init.__dict__[initialization_method]


# pylint: disable-next=invalid-name
def huber_loss(e, d):
    """Huber loss."""
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)


# pylint: disable-next=invalid-name
def mse_loss(e):
    """MSE loss."""
    return e**2 / 2


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly
    Args:
        optimizer: (torch.optim) optimizer
        epoch: (int) current epoch
        total_num_epochs: (int) total number of epochs
        initial_lr: (float) initial learning rate
    """
    learning_rate = initial_lr - (initial_lr * ((epoch - 1) / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def init(module, weight_init, bias_init, gain=1):
    """Init module.
    Args:
        module: (torch.nn) module
        weight_init: (torch.nn) weight init
        bias_init: (torch.nn) bias init
        gain: (float) gain
    Returns:
        module: (torch.nn) module
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    """Clone module for N times."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def get_grad_norm(parameters):
    """Get gradient norm."""
    sum_grad = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        sum_grad += parameter.grad.norm() ** 2
    return math.sqrt(sum_grad)
