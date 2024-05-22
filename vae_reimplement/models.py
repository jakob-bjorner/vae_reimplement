import torch.nn as nn
class CNNDecoder(nn.Module):
    def __init__(self, image_HW, image_channels, prior_dim):
        super(CNNDecoder, self).__init__()
        self.H = image_HW
        self.C = image_channels
        assert self.H == 32, "only works for h w = 32"
        self.prior_dim = prior_dim
        self.expand_prior = nn.Linear(self.prior_dim, self.prior_dim * self.H // 4 * self.H // 4)
        self.tanh = nn.Tanh() # TODO: figure out how to easily try different hyper parameters in the decoder with hydra?
        self.expand_2 = nn.ConvTranspose2d(self.prior_dim, self.C * 2, kernel_size=4,stride=4)
        self.flatten = nn.Flatten()
    def forward(self, z):
        x = self.expand_prior(z).reshape(-1, self.prior_dim, self.H // 4,  self.H // 4)
        x = self.expand_2(self.tanh(x))
        x = self.flatten(x)
        return x
class LinearDecoder(nn.Module):
    def __init__(self, image_HW, image_channels, prior_dim):
        super(LinearDecoder, self).__init__()
        self.H = image_HW
        self.C = image_channels
        assert self.H == 32, "only works for h w = 32"
        self.prior_dim = prior_dim
        self.expand_prior = nn.Linear(self.prior_dim, self.prior_dim * self.H * self.H)
        self.tanh = nn.Tanh()
        self.expand_2 = nn.Linear(self.prior_dim * self.H * self.H, 2 * self.H * self.H * self.C)
    def forward(self, z):
        x = self.expand_prior(z)
        x = self.expand_2(self.tanh(x))
        return x
    

class CNNEncoder(nn.Module):
    def __init__(self, image_HW, image_channels, prior_dim, last_kernel_size=8):
        super(CNNEncoder, self).__init__()
        self.H = image_HW
        self.C = image_channels
        assert self.H % last_kernel_size == 0, f"need the last_kernel_size to divide the image shape {last_kernel_size}, {self.H}" 
        self.layers = nn.Sequential()
        in_channels = self.C
        for i in range(self.H // last_kernel_size - 1):
            self.layers.append(nn.Conv2d(in_channels, in_channels * 2, kernel_size=last_kernel_size+1))
            if i % 2 == 0:
                self.layers.append(nn.ReLU())
            in_channels *= 2
        self.layers.append(nn.Conv2d(in_channels, in_channels*2, kernel_size=last_kernel_size))
        in_channels *= 2
        self.prior_dim = prior_dim
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(in_channels, 2 * self.prior_dim))
    def forward(self, x):
        return self.layers(x)

class LinearEncoder(nn.Module):
    def __init__(self, image_HW, image_channels, prior_dim, last_kernel_size=8):
        super(LinearEncoder, self).__init__()
        self.H = image_HW
        self.C = image_channels
        self.prior_dim = prior_dim
        assert self.H % last_kernel_size == 0, f"need the last_kernel_size to divide the image shape {last_kernel_size}, {self.H}" 
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(self.H * self.H * self.C, self.H * self.H * self.C * 2)
        self.activation = nn.Tanh()
        self.linear2 = nn.Linear(self.H * self.H * self.C * 2, self.prior_dim * 2)
    def forward(self, x):
        return self.linear2(self.activation(self.linear1(self.flatten(x))))