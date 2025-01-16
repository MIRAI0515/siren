import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
from collections import OrderedDict
import argparse
import time

def get_mgrid(sidelen, dim=3):
    """
    Generate a meshgrid for n-dimensional data.
    `sidelen` can be a single integer or a tuple specifying the size of each dimension.
    """
    if isinstance(sidelen, int):
        sidelen = (sidelen,) * dim
    tensors = [torch.linspace(-1, 1, steps=s) for s in sidelen]
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output

def get_volume_tensor(volume_path):
    """
    Load 3D volume data and normalize it to [-1, 1].
    """
    volume = np.load(volume_path)
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))  # Normalize to [0, 1]
    volume = 2 * volume - 1  # Normalize to [-1, 1]
    volume = torch.tensor(volume, dtype=torch.float32)
    return volume

class VolumeFitting(Dataset):
    def __init__(self, volume_path):
        super().__init__()
        volume = get_volume_tensor(volume_path)
        self.pixels = volume.view(-1, 1)
        self.coords = get_mgrid(sidelen=volume.shape, dim=3)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError
        return self.coords, self.pixels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument('--logging_root', type=str, default='/mnt/siren/logs', help='Root for logging.')
    parser.add_argument('--output_dir', type=str, default='/mnt/siren/explore_siren', help='Directory to save outputs.')
    parser.add_argument('--volume_path', type=str, required=True, help='Path to the input volume (numpy .npy file).')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--num_epochs', type=int, default=10000, help='Number of epochs for training.')
    parser.add_argument('--steps_til_summary', type=int, default=200, help='Steps until summary.')

    opt = parser.parse_args()

    volume_shape = np.load(opt.volume_path).shape
    opt.output_size = volume_shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VolumeFitting(volume_path=opt.volume_path)
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)

    print(f"Volume shape: {volume_shape}")

    model = Siren(in_features=3, out_features=1, hidden_features=256, hidden_layers=3, outermost_linear=True).to(device)

    optim = torch.optim.Adam(lr=opt.lr, params=model.parameters())

    os.makedirs(opt.output_dir, exist_ok=True)

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.to(device), ground_truth.to(device)

    for epoch in range(opt.num_epochs):
        model_output = model(model_input)
        loss = ((model_output - ground_truth)**2).mean()

        if epoch % opt.steps_til_summary == 0:
            print(f"Step {epoch}, Loss: {loss.item():.6f}")
            output_volume = model_output.cpu().view(*volume_shape).detach().numpy()
            np.save(os.path.join(opt.output_dir, f"output_step_{epoch}.npy"), output_volume)

        optim.zero_grad()
        loss.backward()
        optim.step()

    torch.save(model.state_dict(), os.path.join(opt.output_dir, 'model.pth'))
