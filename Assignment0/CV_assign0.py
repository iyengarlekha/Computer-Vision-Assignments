import torch
import matplotlib.pyplot as plt 
import math
import torch.nn

def get_covariance(x):
    return x.t().mm(x) / (x.size(dim = 0) - 1)

def whitening_data():

    data_raw = torch.load('assign0_data.py')

    plt.scatter(data_raw[:,0], data_raw[:,1], label = 'Raw_data')
    plt.legend()
    plt.show()
    plt.clf()

    # Center the data
    data_centered = data_raw - data_raw.mean(dim = 0)

    # get Whitening Matrix (W)
    M = get_covariance(data_centered)
    
    d,e = M.symeig(eigenvectors = True)
    D = d.pow(-0.5) * torch.eye(2)
    E = e
    W = E.mm(D)
    
    # Apply Whitening Transform
    data = data_centered.mm(W)

    # Plot the whitened data
    plt.scatter(data[:,0], data[:,1], label = 'Whitened data')
    plt.legend()
    plt.show()
    plt.clf()

    # Check the final covariance
    M = get_covariance(data)
    print M

def simple_neural_net():
    x = torch.arange(-math.pi,math.pi, 0.01)
    y = torch.cos(x)
    
    N= len(x)
    x=x.view(N,1)
    y=y.view(N,1)
    
    H= 10
    D_out = 1
    D_in = 1
    
    # Model definition
    model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.Tanh(),
    torch.nn.Linear(H, D_out),
)
    
    # Loss function (Mean squared error loss)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-3

    # Use Adam optimizer to update weights of the model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    y_untrained  = model(x)
    
    for t in range(1000):
        # Forward pass    
        y_pred = model(x)
        
        # Compute loss
        loss = loss_fn(y_pred, y)
        
        # Before the backward pass, use the optimizer object to zero all of the 
        # gradients for the variables it will update. This is because by default, 
        # gradients are accumulated in buffers( i.e, not overwritten)  whenever 
        # .backward() is called
        optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Step function called on an Optimizer updates parameters
        optimizer.step()
    
    plt.plot(x.detach().numpy(),y_untrained.detach().numpy(), color = 'red', label = 'Untrained function')    
    plt.plot(x.detach().numpy(),y.detach().numpy(), color = 'green', label = 'True function')
    plt.plot(x.detach().numpy(),y_pred.detach().numpy(), color = 'blue', label = 'Trained function')
    plt.legend()
    plt.show()
    
def main():
    whitening_data()
    simple_neural_net()

if __name__ == '__main__':
    main()