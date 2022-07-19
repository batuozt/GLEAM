"""
Implementations of different CNNs

by Christopher M. Sandino (sandino@stanford.edu), 2019.

"""

from torch import nn
from collections import OrderedDict

from ss_recon.utils.transforms import center_crop

class ConvBlock(nn.Module):
    """
    A 3D Convolutional Block that consists of Norm -> ReLU -> Dropout -> Conv

    Based on implementation described by:
        K He, et al. "Identity Mappings in Deep Residual Networks" arXiv:1603.05027
    """

    def __init__(
        self,
        in_chans,
        out_chans,
        kernel_size,
        drop_prob,
        #conv_type="conv3d",
        act_type="relu",
        #norm_type="none",
    ):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        conv_type="conv3d"
        norm_type="none"
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        # Define choices for each layer in ConvBlock
        normalizations = nn.ModuleDict(
            [
                ["none", nn.Identity()],
                ["instance", nn.InstanceNorm3d(in_chans, affine=False)],
                ["batch", nn.BatchNorm3d(in_chans, affine=False)],
            ]
        )
        activations = nn.ModuleDict([["relu", nn.ReLU()], ["leaky_relu", nn.LeakyReLU()], ["none", nn.Identity()]])
        dropout = nn.Dropout3d(p=drop_prob, inplace=True)

        # Note: don't use ModuleDict here. Otherwise, the parameters for the un-selected
        # convolution type will still be initialized and added to model.parameters()
        convolution = nn.Conv3d(in_chans, out_chans, kernel_size=kernel_size,bias=False,padding=1,padding_mode='replicate')
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        padding_layer=nn.ReplicationPad3d    
        pad_params=(ka,kb,ka,kb,ka,kb)
            
        # Define forward pass
        self.layers = nn.Sequential(
            #normalizations[norm_type],
            #dropout,
            #padding_layer(pad_params),
            convolution,
            activations[act_type],
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, depth, width, height]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, depth, width, height]
        """
        return self.layers(input)

    def __repr__(self):
        return (
            f"ConvBlock3D(in_chans={self.in_chans}, out_chans={self.out_chans}, "
            f"drop_prob={self.drop_prob})"
        )


class ResBlock(nn.Module):
    """
    A ResNet block that consists of two convolutional layers followed by a residual connection.
    """

    def __init__(self, in_chans, out_chans, kernel_size, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.layers = nn.Sequential(
            ConvBlock(in_chans, out_chans, kernel_size, drop_prob),
            ConvBlock(out_chans, out_chans, kernel_size, drop_prob),
        )

        if in_chans != out_chans:
            self.resample = nn.Conv3d(in_chans, out_chans, kernel_size=1)
        else:
            self.resample = nn.Identity()

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, depth, width, height]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, depth, width, height]
        """

        # To have a residual connection, number of inputs must be equal to outputs
        shortcut = self.resample(input)

        return self.layers(input) + shortcut


class ResNet(nn.Module):
    """
    Prototype for 3D ResNet architecture
    """

    def __init__(
        self,
        num_resblocks,
        in_chans,
        chans,
        kernel_size,
        drop_prob,
        circular_pad=True,
    ):
        """ """
        super().__init__()

        self.circular_pad = circular_pad
        self.pad_size = 2 * num_resblocks + 1

        # Declare ResBlock layers
        self.res_blocks = nn.ModuleList([ResBlock(in_chans, chans, kernel_size, drop_prob)])
        for _ in range(num_resblocks - 1):
            self.res_blocks += [ResBlock(chans, chans, kernel_size, drop_prob)]

        # Declare final conv layer (down-sample to original in_chans)
        self.final_layer = nn.Conv3d(chans, in_chans, kernel_size=kernel_size, padding=1)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, depth, width, height]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.in_chans, depth, width, height]
        """

        orig_shape = input.shape
        if self.circular_pad:
            input = nn.functional.pad(input, (0, 0, 0, 0) + 2 * (self.pad_size,), mode="circular")
            # input = nn.functional.pad(input, 4*(self.pad_size,) + (0,0), mode='replicate')

        # Perform forward pass through the network
        output = input
        for res_block in self.res_blocks:
            output = res_block(output)
        output = self.final_layer(output) + input

        return center_crop(output, orig_shape)
    
class RevNet(nn.Module):
    """
    Invertible residual networks used in MEL: https://arxiv.org/abs/2103.04003
    """
    def __init__(self, dims, num_channels=32, kernel_size=3, T=5, num_layers=3,device='cpu'):
        super(RevNet, self).__init__()
        
        conv = lambda in_channels, out_channels, filter_size: ConvSame(in_channels, out_channels, filter_size, dims)
        #conv = lambda in_channels, out_channels, filter_size, drop_prob, act_type: ConvBlock(in_channels, out_channels, filter_size, drop_prob, act_type)
        self.dims = dims
        self.device = device
        layer_dict = OrderedDict()
        #layer_dict['conv1'] = conv(2,num_channels,kernel_size,0.,act_type="relu")
        layer_dict['conv1'] = conv(2,num_channels,kernel_size)
        layer_dict['relu1'] = nn.ReLU()
        for i in range(num_layers-2):
            #layer_dict[f'conv{i+2}'] = conv(num_channels, num_channels, kernel_size,0.,act_type="relu")
            layer_dict[f'conv{i+2}'] = conv(num_channels, num_channels, kernel_size)
            layer_dict[f'relu{i+2}'] = nn.ReLU()
        layer_dict[f'conv{num_layers}'] = conv(num_channels,2,kernel_size)
        #layer_dict[f'conv{num_layers}'] = conv(num_channels,2,kernel_size,0.,act_type="none")
        
        self.model = nn.Sequential(layer_dict).to(self.device)
        self.T = T
        #import pdb; pdb.set_trace()
        
    def forward(self,x):
        return x + self.step(x)
    
    def step(self,x):
        #x = x.permute(0,3,1,2)
        y = self.model(x)
        #y = y.permute(0,2,3,1)
        # reshape (batch,channel=2,x,y) -> (batch,x,y,channel=2)
        return y
    
    def reverse(self, x):
        z = x
        for _ in range(self.T):
            z = x - self.step(z)
        return z
    
class ConvSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dims, bias=False):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        if dims == 3:
            padding_layer=nn.ReplicationPad3d
            conv_layer=nn.Conv3d
            pad_params=(ka,kb,ka,kb,ka,kb)
        elif dims == 2:
            padding_layer=nn.ReflectionPad2d
            conv_layer=nn.Conv2d
            pad_params=(ka,kb,ka,kb)
            
        conv_params={'in_channels':in_channels, 'out_channels':out_channels, 'kernel_size':kernel_size, 'bias':bias}
        self.net = nn.Sequential(
            padding_layer(pad_params),
            conv_layer(**conv_params)
        )
    def forward(self, x):
        return self.net(x)