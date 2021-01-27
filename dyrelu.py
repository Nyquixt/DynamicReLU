import torch
import torch.nn as nn

class DyReLU(nn.Module):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # buffer won't be trained by the optimizer
        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        # alphas and betas, a1 = 1.0, a2 = b1 = b2 = 0.0
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float())

    def get_relu_coefs(self, x):
        theta = torch.mean(x, dim=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, dim=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1 # normalization
        return theta

    def forward(self, x):
        raise NotImplementedError


class DyReLUA(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUA, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)
        
        relu_coefs = theta.view(-1, 2*self.k) * self.lambdas + self.init_v
        # BxCxL -> LxCxBx1
        x_perm = x.transpose(0, -1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :self.k] + relu_coefs[:, self.k:]
        # LxCxBx2 -> BxCxL
        result = torch.max(output, dim=-1)[0].transpose(0, -1)

        return result


class DyReLUB(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUB, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k*channels)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v

        if self.conv_type == '1d':
            # BxCxL -> LxBxCx1
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # LxBxCx2 -> BxCxL
            result = torch.max(output, dim=-1)[0].permute(1, 2, 0)

        elif self.conv_type == '2d':
            # BxCxHxW -> HxWxBxCx1
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # HxWxBxCx2 -> BxCxHxW
            result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

        return result

class DyReLUC(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d', temp=10):
        super(DyReLUC, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k*channels)
        self.temp = temp
        if conv_type == '1d':
            self.conv1x1 = nn.Conv1d(channels, 1, 1)
        if conv_type == '2d': 
            self.conv1x1 = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v

        if self.conv_type == '1d':
            b, c, l = x.size()
            gamma = l/3
            #BxCxL
            x_prime = self.conv1x1(x)
            x_prime = gamma * torch.softmax(x_prime/self.temp, dim=-1)
            x_prime = torch.min(x_prime, torch.ones(b, c, l))
            # BxCxL -> LxBxCx1
            x_perm = x.permute(2, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # LxBxCx2 -> BxCxL
            result = torch.max(output, dim=-1)[0].permute(1, 2, 0)

        elif self.conv_type == '2d':
            b, c, h, w = x.size()
            gamma = (h*w)/3
            #BxCxHW
            x_prime = self.conv1x1(x).view(b, 1, -1)
            x_prime = gamma * torch.softmax(x_prime/self.temp, dim=-1)
            #BxCxHW -> #BxCxHxW
            x_prime = torch.min(x_prime, torch.ones(b, c, h*w)).view(-1, c, h, w)
            
            # BxCxHxW -> HxWxBxCx1
            x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
            output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
            # HxWxBxCx2 -> BxCxHxW
            result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

        return x_prime * result

def test():
    x = torch.randn(3, 10, 32, 32) # NxCxHxW
    reluA = DyReLUA(10)
    reluB = DyReLUB(10)
    reluC = DyReLUC(10)
    yA = reluA(x)
    yB = reluB(x)
    yC = reluC(x)
    print(yA.shape, yB.shape, yC.shape)

    x = torch.randn(3, 10, 32) # NxCxL
    reluA = DyReLUA(10, conv_type='1d')
    reluB = DyReLUB(10, conv_type='1d')
    reluC = DyReLUC(10, conv_type='1d')
    yA = reluA(x)
    yB = reluB(x)
    yC = reluC(x)
    print(yA.shape, yB.shape, yC.shape)

# test()