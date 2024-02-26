"""PSE block
A variant of Squeeze-and-Excitation (SE) block

Created by Yiran Wang
"""
import torch
import torch.nn as nn


class PermuteSE(nn.Module):
    """
    BCN -> Permute -> SE
    input shape: B, C_in, H, W
    output shape: B, C_out, 1, 1
    parallel style block: output = (output of the other block) * PermuteSE(input)
    """

    PSE_idx = 0

    def __init__(self, c_in, c_out, g=1, img_size_list=None):
        """
        Args:
            c_in: input channel
            c_out: output channel
            img_size_list: a list of sizes of the images that input to PSE block, e.g. [224, 56,] or [(224, 112), ... ]
        """
        super(PermuteSE, self).__init__()
        assert isinstance(img_size_list, list), 'You need to provide a list of image sizes for PermuteSE'

        img_size = img_size_list[PermuteSE.PSE_idx]
        PermuteSE.PSE_idx += 1

        if isinstance(img_size, tuple):
            h = img_size[0]
            w = img_size[1]
        else:
            h = w = img_size
        self.bcn = BatchChannelNorm(c_in)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(c_in, c_out, 1, 1, 0, groups=g, bias=False)  # channel attention
        self.conv2 = nn.Conv2d(h, c_out, 1, 1, 0, groups=g, bias=False)  # height attention
        self.conv3 = nn.Conv2d(w, c_out, 1, 1, 0, groups=g, bias=False)  # width attention
        self.act = nn.Sigmoid()

    def forward(self, x):
        # print('PSE input shape:', x.shape)
        x = self.bcn(x)
        x1 = x  # x1: B, C, H, W
        x2 = x.permute(0, 2, 1, 3)  # x2: B, H, C, W
        x3 = x.permute(0, 3, 2, 1)  # x3: B, W, H, C

        x1 = self.avg_pool(x1)
        x1 = self.conv1(x1)
        x1 = self.act(x1)
        # return x1

        x2 = self.avg_pool(x2)
        x2 = self.conv2(x2)
        x2 = self.act(x2)

        x3 = self.avg_pool(x3)
        x3 = self.conv3(x3)
        x3 = self.act(x3)

        out = x1 + x2 + x3
        # print('PSE output shape:', out.shape)
        return out


class PermuteSEV2(nn.Module):
    """
    BCN -> Permute -> SE
    input shape: B, C, H, W
    output shape: B, C, H, W
    input shape == output shape
    Serial style block: output = PermuteSEV2(input)
    """

    PSE_idx = 0

    def __init__(self, c_in, c_out, g=1, img_size_list=None):
        """
        Args:
            c_in: input channel
            c_out: output channel
            img_size_list: a list of sizes of the images that input to PSE block, e.g. [224, 56,] or [(224, 112), ... ]
        """
        super(PermuteSEV2, self).__init__()
        assert isinstance(img_size_list, list), 'You need to provide a list of image sizes for PermuteSE'
        assert c_in == c_out, 'c_in and c_out must be equal in PermuteSEV2'

        img_size = img_size_list[PermuteSEV2.PSE_idx]
        PermuteSEV2.PSE_idx += 1
        if isinstance(img_size, tuple):
            h = img_size[0]
            w = img_size[1]
        else:
            h = w = img_size
        self.bcn = BatchChannelNorm(c_in)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(c_in, c_out, 1, 1, 0, groups=g, bias=False)  # channel attention
        self.conv2 = nn.Conv2d(h, c_out, 1, 1, 0, groups=g, bias=False)  # height attention
        self.conv3 = nn.Conv2d(w, c_out, 1, 1, 0, groups=g, bias=False)  # width attention
        self.act = nn.Sigmoid()

    def forward(self, x):
        # print('PSE input shape:', x.shape)
        x = self.bcn(x)
        x1 = x  # x1: B, C, H, W
        x2 = x.permute(0, 2, 1, 3)  # x2: B, H, C, W
        x3 = x.permute(0, 3, 2, 1)  # x3: B, W, H, C

        x1 = self.avg_pool(x1)
        x1 = self.conv1(x1)
        x1 = self.act(x1)
        # return x1

        x2 = self.avg_pool(x2)
        x2 = self.conv2(x2)
        x2 = self.act(x2)

        x3 = self.avg_pool(x3)
        x3 = self.conv3(x3)
        x3 = self.act(x3)

        out = (x1 + x2 + x3) * x
        # print('PSE output shape:', out.shape)
        return out


class PermuteSEV3(nn.Module):
    """
    Permute -> SE (No BCN)
    input shape: B, C, H, W
    output shape: B, C, H, W
    input shape == output shape
    Serial style block: output = PermuteSEV2(input)
    """

    PSE_idx = 0

    def __init__(self, c_in, c_out, g=1, img_size_list=None):
        """
        Args:
            c_in: input channel
            c_out: output channel
            img_size_list: a list of sizes of the images that input to PSE block, e.g. [224, 56,] or [(224, 112), ... ]
        """
        super(PermuteSEV3, self).__init__()
        assert isinstance(img_size_list, list), 'You need to provide a list of image sizes for PermuteSE'
        assert c_in == c_out, 'c_in and c_out must be equal in PermuteSEV2'

        img_size = img_size_list[PermuteSEV3.PSE_idx]
        PermuteSEV3.PSE_idx += 1
        if isinstance(img_size, tuple):
            h = img_size[0]
            w = img_size[1]
        else:
            h = w = img_size
        # self.bcn = BatchChannelNorm(c_in)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(c_in, c_out, 1, 1, 0, groups=g, bias=False)  # channel attention
        self.conv2 = nn.Conv2d(h, c_out, 1, 1, 0, groups=g, bias=False)  # height attention
        self.conv3 = nn.Conv2d(w, c_out, 1, 1, 0, groups=g, bias=False)  # width attention
        self.act = nn.Sigmoid()

    def forward(self, x):
        # print('PSE input shape:', x.shape)
        # x = self.bcn(x)
        x1 = x  # x1: B, C, H, W
        x2 = x.permute(0, 2, 1, 3)  # x2: B, H, C, W
        x3 = x.permute(0, 3, 2, 1)  # x3: B, W, H, C

        x1 = self.avg_pool(x1)
        x1 = self.conv1(x1)
        x1 = self.act(x1)
        # return x1

        x2 = self.avg_pool(x2)
        x2 = self.conv2(x2)
        x2 = self.act(x2)

        x3 = self.avg_pool(x3)
        x3 = self.conv3(x3)
        x3 = self.act(x3)

        out = (x1 + x2 + x3) * x
        # print('PSE output shape:', out.shape)
        return out

class BatchChannelNorm(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, momentum=0.9):
        super(BatchChannelNorm, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.momentum = momentum
        self.Batchh = BatchNormm2D(self.num_channels, epsilon=self.epsilon)
        self.layeer = LayerNormm2D(self.num_channels, epsilon=self.epsilon)
        # The BCN variable to be learnt
        self.BCN_var = nn.Parameter(torch.ones(self.num_channels))
        # Gamma and Beta for rescaling
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        X = self.Batchh(x)
        Y = self.layeer(x)
        out = self.BCN_var.view([1, self.num_channels, 1, 1]) * X + (
                1 - self.BCN_var.view([1, self.num_channels, 1, 1])) * Y
        out = self.gamma.view([1, self.num_channels, 1, 1]) * out + self.beta.view([1, self.num_channels, 1, 1])
        return out


class BatchNormm2D(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, momentum=0.9, rescale=True):
        super(BatchNormm2D, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.momentum = momentum
        self.rescale = rescale

        if (self.rescale == True):
            # define parameters gamma, beta which are learnable
            # dimension of gamma and beta should be (num_channels) ie its a one dimensional vector
            # initializing gamma as ones vector and beta as zeros vector (implies no scaling/shifting at the start)
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))
        # define parameters running mean and variance which is not learnable
        # keep track of mean and variance(but donot learn them), momentum is used which weighs current batch-mean and
        # variance with the running mean and variance using (momentum*runningmean+(1-momentum)*currentmean)
        self.register_buffer('runningmean', torch.zeros(num_channels))
        self.register_buffer('runningvar', torch.ones(num_channels))

    def forward(self, x):
        assert x.shape[1] == self.num_channels
        assert len(x.shape) == 4  # 4 because (batchsize, numchannels, height, width)

        if (self.training):
            # calculate mean and variance along the dimensions other than the channel dimension
            # variance calculation is using the biased formula during training
            variance = torch.var(x, dim=[0, 2, 3], unbiased=False)
            mean = torch.mean(x, dim=[0, 2, 3])
            self.runningmean = (1 - self.momentum) * mean + (self.momentum) * self.runningmean
            self.runningvar = (1 - self.momentum) * variance + (self.momentum) * self.runningvar
            out = (x - mean.view([1, self.num_channels, 1, 1])) / torch.sqrt(
                variance.view([1, self.num_channels, 1, 1]) + self.epsilon)
        else:
            m = x.shape[0] * x.shape[2] * x.shape[3]
            out = (x - self.runningmean.view([1, self.num_channels, 1, 1])) / torch.sqrt(
                (m / (m - 1)) * self.runningvar.view([1, self.num_channels, 1, 1]) + self.epsilon)
            # during testing just use the running mean and (UnBiased) variance
        if (self.rescale == True):
            return out


class LayerNormm2D(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5):
        super(LayerNormm2D, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon

        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        assert list(x.shape)[1] == self.num_channels
        assert len(x.shape) == 4  # 4 because len((batchsize, numchannels, height, width)) = 4
        variance, mean = torch.var(x, dim=[1, 2, 3], unbiased=False), torch.mean(x, dim=[1, 2, 3])

        out = (x - mean.view([-1, 1, 1, 1])) / torch.sqrt(variance.view([-1, 1, 1, 1]) + self.epsilon)
        return out
