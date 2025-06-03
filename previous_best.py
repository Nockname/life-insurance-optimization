# ---- Find Best Naive Premium Policies ----

import torch
import model
import matplotlib.pyplot as plt
import train
from constants import *



# constant premium policy
def train_const(premium_input = 10.0, n_iters=500000, lr=2, weight = 1.0, balance_activation_function=torch.nn.ELU(5.0), premiums_modification_function = torch.nn.ReLU()):

    premium_input = torch.tensor(premium_input)
    premium_input.requires_grad_()
    
    optimizer = torch.optim.Adam([premium_input], lr=lr)
        
    losses = []
    best = (torch.empty((20,)), float('inf'))
    
    for i in range(n_iters):
        
        long_input = premium_input.repeat(YEARS)
        
        # gradient descent step
        optimizer.zero_grad()

        loss_value = model.smooth_loss(long_input, weight, balance_activation_function=balance_activation_function, premiums_modification_function=premiums_modification_function)

        losses.append((loss_value.item(), model.positive_balance_years(long_input, premiums_modification_function=premiums_modification_function)))

        if loss_value < best[1]:
            best = (long_input.clone().detach(), loss_value)

        loss_value.backward()
        optimizer.step()
        
        
        train.print_partial_results(losses, long_input, n_iters, i, premiums_modification_function)

    return losses, best


# early premium policy
def train_early(premium_input = 10.0, n_iters=500000, lr=2, weight = 1.0, balance_activation_function=torch.nn.ELU(5.0), premiums_modification_function = torch.nn.ReLU()):
    
    zeros = torch.full((YEARS - 1,), -float('inf'))

    premium_input = torch.tensor(premium_input)
    premium_input.requires_grad_()
    
    optimizer = torch.optim.Adam([premium_input], lr=lr)
        
    losses = []
    best = (torch.empty((20,)), float('inf'))
    
    for i in range(n_iters):
        
        long_input = torch.cat((premium_input.repeat(1), zeros), dim=0)
        # gradient descent step
        optimizer.zero_grad()

        loss_value = model.smooth_loss(long_input, weight, balance_activation_function=balance_activation_function, premiums_modification_function=premiums_modification_function)

        losses.append((loss_value.item(), model.positive_balance_years(long_input, premiums_modification_function=premiums_modification_function)))

        if loss_value < best[1]:
            best = (long_input.clone().detach(), loss_value)

        loss_value.backward()
        optimizer.step()
        
        
        train.print_partial_results(losses, long_input, n_iters, i, premiums_modification_function)

    return losses, best


torch.manual_seed(0)
preprocessing = torch.exp

losses, best_const = train_const(premium_input=10.0, n_iters=3000, lr=0.001, weight=0.5, premiums_modification_function=preprocessing)
losses, best_early = train_early(premium_input=10.0, n_iters=3000, lr=0.001, weight=0.5, premiums_modification_function=preprocessing)

# print("\n\nConstant Premiums:")
# train.print_results(losses, best_const, premiums_modification_function=preprocessing)
# print("\n\nEarly Premiums:")
# train.print_results(losses, best_early, premiums_modification_function=preprocessing)

best_const_premiums = torch.tensor([round(float(preprocessing(p).item()), 2) for p in best_const[0]])
best_early_premiums = torch.tensor([round(float(preprocessing(p).item()), 2) for p in best_early[0]])
# print(f"Best Constant Premiums: {best_const_premiums}")
# print(f"Best Early Premiums: {best_early_premiums}")