import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        # Capas del bloque residual
        self.main = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        # Conexion de salto residual
        return x + self.main(x)


class Generator(nn.Module):
    def __init__(self, num_instrumentos=5, num_generos=15, num_bloques_residuales=6):
        super(Generator, self).__init__()
        
        # unir instrumentos con generos
        input_dim = num_instrumentos + num_generos
        
        # primera capa convolucional
        capas = [
            nn.Conv2d(input_dim, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        ]
        
        # capas de reduccion
        curr_dim = 64
        for _ in range(2):
            capas += [
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            ]
            curr_dim = curr_dim * 2
            
        # agregar bloques residuales intermedias
        for _ in range(num_bloques_residuales):
            capas += [ResidualBlock(dim=curr_dim)]
            
        # capas de expansión
        for _ in range(2):
            capas += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            ]
            curr_dim = curr_dim // 2
            
        # capa final convolucional
        capas += [
            nn.Conv2d(curr_dim, num_instrumentos, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh() 
        ]
        
        self.main = nn.Sequential(*capas)

    def forward(self, x, c):
        # Reestructurar dimensiones del genero
        c = c.view(c.size(0), c.size(1), 1, 1)
        # Expandir tamaño del genero
        c = c.repeat(1, 1, x.size(2), x.size(3))
        # Juntar matrices por canales
        inputs = torch.cat([x, c], dim=1)
        return self.main(inputs)


class Discriminator(nn.Module):
    def __init__(self, num_instrumentos=5, num_generos=15):
        super(Discriminator, self).__init__()        
        # capa inicial del discriminador
        capas = [
            nn.Conv2d(num_instrumentos, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01)
        ]        
        # capas ocultas avanzadas
        curr_dim = 64
        for _ in range(4):
            capas += [
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.01)
            ]
            curr_dim = curr_dim * 2            
        self.main = nn.Sequential(*capas)        
        # salida de validez
        self.conv_validez = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)        
        # salida de clasificación
        self.conv_clasificacion = nn.Conv2d(curr_dim, num_generos, kernel_size=4, stride=1, padding=0, bias=False)

    def forward(self, x):
        # Extraer caracteristicas del tensor
        h = self.main(x)        
        # Calcular salidas finales
        out_validez = self.conv_validez(h)
        out_clasificacion = self.conv_clasificacion(h)        
        return out_validez, out_clasificacion.view(out_clasificacion.size(0), out_clasificacion.size(1))