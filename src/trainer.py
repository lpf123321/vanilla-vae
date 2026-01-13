import torch
import os
from torch.nn import functional as F
from torchvision.utils import save_image

def loss_function(recon_x, x, mu, logvar, kld_weight=1.0):
    # BCE Loss is the correct likelihood for MNIST (values in [0, 1])
    # reduction='sum' sums over the batch and pixels
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KLD term
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Standard VAE ELBO = Reconstruction + KLD
    # usually we optimize the sum over the batch
    return (BCE + kld_weight * KLD)

def train(model, device, train_loader, optimizer, epoch, log_interval, kld_weight):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        
        # Calculate loss (sum over batch)
        loss = loss_function(recon_batch, data, mu, logvar, kld_weight)
        
        # Backward
        loss.backward()
        
        # Normalize loss for logging only
        train_loss += loss.item()
        
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    avg_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))

def test(model, device, test_loader, kld_weight):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar, kld_weight).item()
            
            if i == 0:
                n = min(data.size(0), 8)
                # Dynamic shape handling for visualization
                # recon_batch is already (B, C, H, W)
                comparison = torch.cat([data[:n], recon_batch[:n]])
                if not os.path.exists('results'):
                    os.makedirs('results')
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(i) + '.png', nrow=n)

    avg_test_loss = test_loss / len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(avg_test_loss))
