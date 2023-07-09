import torch
import torch.nn as nn

class ResBlock256(nn.Module):
    def __init__(self):
        super(ResBlock256, self).__init__()
        
        self.proc = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),
        )
        self._init_weights()
        
    def _init_weights(module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            module.weight.data.normal_(mean=0.0, std=0.02)
            module.bias.data.fill_(0)
    
    def forward(self, x_in):
        x = self.proc(x_in)
        return x + x_in

class CycleGANGenerator(nn.Module):
    def __init__(self):
        super(CycleGANGenerator, self).__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d((3, 3, 3, 3)),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            ResBlock256(),
            nn.ReLU(True),
            ResBlock256(),
            nn.ReLU(True),
            ResBlock256(),
            nn.ReLU(True),
            ResBlock256(),
            nn.ReLU(True),
            ResBlock256(),
            nn.ReLU(True),
            ResBlock256(),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            nn.ReflectionPad2d((3, 3, 3, 3)),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1),
            nn.Tanh()
        )
        self._init_weights()
        
    def _init_weights(module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            module.weight.data.normal_(mean=0.0, std=0.02)
            module.bias.data.fill_(0)
        
    def forward(self, x_in):
        return self.net(x_in)
    
class CycleGANDiscriminator(nn.Module):
    def __init__(self):
        super(CycleGANDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1),
        )
        self._init_weights()
        
    def _init_weights(module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            module.weight.data.normal_(mean=0.0, std=0.02)
            module.bias.data.fill_(0)
        
    def forward(self, x_in):
        return self.net(x_in)