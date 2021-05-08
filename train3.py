import os
import json
import argparse
import torch
from torch.nn import functional as F
import random
import torch.optim as optim
from dataset2_2 import AudioDataloader
from model_vc3 import Generator
import numpy as np
from hparams import hparams as hp

       
def save_checkpoint(device, model, optimizer, checkpoint_dir, epoch):
    checkpoint_path = os.path.join(
        checkpoint_dir, "checkpoint_original_step{:06d}.pth".format(epoch))
    optimizer_state = optimizer.state_dict()
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer_state,
        "epoch": epoch
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def load_checkpoint(path, model, device, optimizer, reset_optimizer=False):
    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint['epoch'] 
    return epoch

def train(args, model, device, train_loader, optimizer, epoch, sigma=1.0):
    model.train()
    train_loss = 0

    for batch_idx, (m, e) in enumerate(train_loader.loader()):
        m = m.to(device)
        e = e.to(device)
        
        model.zero_grad()

        y, z1, vq_loss, perplexity = model(m, e, e)
        z2, _, _ = model(m, e)

        recon_loss = F.mse_loss(m, y)
        content_loss = F.l1_loss(z1, z2)

        loss = vq_loss + recon_loss + content_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(m)

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(m), train_loader.data_length(),
                100. * batch_idx / len(train_loader), loss.item()))
        
    train_loss /= train_loader.data_length()
    print('\nTrain set: Average loss: {:.4f}\n'.format(train_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or run some neural net')
    parser.add_argument('-d', '--data', type=str, default='./data', help='dataset directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='The path to checkpoint')
    parser.add_argument('--epochs', type=int, default=600,
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    data_path = args.data

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

    torch.autograd.set_detect_anomaly(True)
    
    train_loader = AudioDataloader('data_32')

    model = Generator().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    current_epoch = 0
    if args.checkpoint:
        current_epoch = load_checkpoint(args.checkpoint, model, device, optimizer)
    
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(current_epoch + 1, args.epochs + 1):
        print(f'epoch {epoch}')
        train(args, model, device, train_loader, optimizer, epoch)

        exit()

        if epoch % 10 == 0:
            save_checkpoint(device, model, optimizer, checkpoint_dir, epoch)
