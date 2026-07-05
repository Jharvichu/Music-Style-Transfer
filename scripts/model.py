import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    # bloque residual con dilatacion
    def __init__(self, dim, dilation=1):
        super(ResidualBlock, self).__init__()
        pad = dilation
        self.main = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=pad, dilation=dilation, bias=False),
            nn.InstanceNorm2d(dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.InstanceNorm2d(dim, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    # generador cyclegan sin condicionales
    def __init__(self, num_instrumentos=5, num_bloques_residuales=6):
        super(Generator, self).__init__()
        
        input_dim = num_instrumentos 
        
        capas = [
            nn.Conv2d(input_dim, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        ]
        
        curr_dim = 64
        for _ in range(2):
            capas += [
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            ]
            curr_dim = curr_dim*2
            
        dilataciones = [1, 2, 4, 1, 2, 4] if num_bloques_residuales == 6 else [1] * num_bloques_residuales
        for dilation in dilataciones:
            capas += [ResidualBlock(dim=curr_dim, dilation=dilation)]
            
        for _ in range(2):
            capas += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(curr_dim, curr_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            ]
            curr_dim = curr_dim//2
            
        capas += [
            nn.Conv2d(curr_dim, num_instrumentos, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid() 
        ]
        
        self.main = nn.Sequential(*capas)

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    # discriminador patchgan de un dominio
    def __init__(self, num_instrumentos=5):
        super(Discriminator, self).__init__()
        
        capas = [
            nn.Conv2d(num_instrumentos, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True) 
        ]
        
        curr_dim = 64
        for _ in range(3): 
            capas += [
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(curr_dim * 2, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            curr_dim = curr_dim * 2
            
        self.main = nn.Sequential(*capas)
        self.conv_validez = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        h = self.main(x)
        return self.conv_validez(h)