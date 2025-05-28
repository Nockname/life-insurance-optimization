import torch
import matplotlib.pyplot as plt
import model

def train(premiums_input = torch.rand((20,)) * 50000.0, n_iters=500000, lr=2, weight = 1.0, balance_activation_function=torch.nn.ELU(5.0), premiums_modification_function = torch.nn.ReLU()):

    premiums_input.requires_grad_()
    optimizer = torch.optim.Adam([premiums_input], lr=lr)
    losses = []
    best = (torch.empty((20,)), float('inf'))
    
    for i in range(n_iters):
        
        # gradient descent step
        optimizer.zero_grad()

        loss_value = model.smooth_loss(premiums_input, weight, balance_activation_function=balance_activation_function, premiums_modification_function=premiums_modification_function)

        losses.append((loss_value.item(), model.positive_balance_years(premiums_input, premiums_modification_function=premiums_modification_function)))

        if loss_value < best[1]:
            best = (premiums_input.clone().detach(), loss_value)

        loss_value.backward()
        optimizer.step()
    
        # print progress bar
        progress = int(50 * (i + 1) / n_iters)
        bar = '[' + '#' * progress + '-' * (50 - progress) + ']'
        print(f"\rProgress: {bar} {100 * (i + 1) / n_iters:.2f}%", end='', flush=True)
        
        # print results every 1% of iterations
        if i % (n_iters / 100) == 0:
            rounded_loss = round(losses[-1][0], 4)
            pos_years = losses[-1][1]
            rounded_premiums = [round(float(premiums_modification_function(p).item()), 2) for p in premiums_input]
            
            # Format: Loss (12 chars, right), Years (3 chars, right), Premiums (each 10 chars, right)
            premiums_str = "[" + ", ".join(f"{p:10.2f}" for p in rounded_premiums) + "]"
            print(f"\rIteration {i:6d}: Loss = {rounded_loss:12.4f}, Positive Balance Years = {pos_years:3d}, Premiums = {premiums_str}")

    return losses, best

def print_results(losses, best, premiums_modification_function):
    print("\n\nTraining complete.")
    print(f"Best Loss: {best[1]:.4f}, with Positive Balance Years: {model.positive_balance_years(best[0], premiums_modification_function=premiums_modification_function)}")
    print("Best Premiums:", [round(float(premiums_modification_function(p).item()), 2) for p in best[0]])

def plot_losses(losses):
    plt.plot([x[0] for x in losses])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Curve During Gradient Descent")
    plt.show()

    plt.plot([x[1] for x in losses])
    plt.xlabel("Iteration")
    plt.ylabel("Positive Balance Years")
    plt.title("Positive Balance Years During Gradient Descent")
    plt.show()


if __name__ == "__main__":

    # Set the random seed for reproducibility
    torch.manual_seed(0)
    preprocessing = torch.exp

    losses, best = train(premiums_input= torch.rand((20,)) * 7 + 7, n_iters=1000000, lr=0.001, weight = 0.5, premiums_modification_function=preprocessing)
    
    print_results(losses, best, premiums_modification_function=preprocessing)
    plot_losses(losses[1250:])
