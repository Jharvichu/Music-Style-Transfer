import torch
import torch.nn as nn
import torch.optim as optim
import os
import itertools
from torch.utils.data import DataLoader

from scripts.dataset import MusicDatasetCycleGAN
from scripts.model import Generator, Discriminator

verde = "\033[1;32m"
turquesa = "\033[1;36m"
rojo = "\033[1;31m"
amarillo = "\033[1;33m"
fin = "\033[0m"

# penalizacion l1 para evitar matrices vacias
def perdida_l1_ponderada(reconstruido, real, penalty_nota=50.0):
    pesos = torch.where(real > 0.05, penalty_nota, 1.0)
    error = torch.abs(reconstruido - real) * pesos
    return error.mean()

def entrenamiento_cyclegan():
    print(f"\n{turquesa}Iniciando entrenamiento cyclegan{fin}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware: {device}")
    
    # parametros de ejecucion local
    epoca_inicial = 0  
    epocas_totales = 100
    batch_size = 4     
    
    # hiperparametros de cyclegan
    lambda_cycle = 10.0
    lambda_identity = 5.0 
    
    genero_A = 'Rock'
    genero_B = 'Jazz'
    
    print("Cargando dataset lpd_5")
    dataset = MusicDatasetCycleGAN('datasets_clasificados/lpd_5', genero_A, genero_B, secuencia_longitud=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # instanciar modelos generadores y discriminadores
    G_A2B = Generator(num_instrumentos=5).to(device) 
    G_B2A = Generator(num_instrumentos=5).to(device) 
    D_A = Discriminator(num_instrumentos=5).to(device) 
    D_B = Discriminator(num_instrumentos=5).to(device) 
    
    # cargar pesos guardados previamente
    if epoca_inicial > 0:
        ruta_G_A2B = f'checkpoints_cyclegan/G_A2B_epoca_{epoca_inicial}.pth'
        ruta_G_B2A = f'checkpoints_cyclegan/G_B2A_epoca_{epoca_inicial}.pth'
        ruta_D_A = f'checkpoints_cyclegan/D_A_epoca_{epoca_inicial}.pth'
        ruta_D_B = f'checkpoints_cyclegan/D_B_epoca_{epoca_inicial}.pth'
        
        if os.path.exists(ruta_G_A2B):
            print(f"{amarillo}Cargando pesos de epoca {epoca_inicial}{fin}")
            G_A2B.load_state_dict(torch.load(ruta_G_A2B, map_location=device))
            G_B2A.load_state_dict(torch.load(ruta_G_B2A, map_location=device))
            D_A.load_state_dict(torch.load(ruta_D_A, map_location=device))
            D_B.load_state_dict(torch.load(ruta_D_B, map_location=device))
        else:
            print(f"{rojo}error checkpoints no encontrados{fin}")
            return
            
    # funcion de perdida mse
    criterion_GAN = nn.MSELoss() 
    
    optimizer_G = optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    os.makedirs('checkpoints_cyclegan', exist_ok=True) 

    print(f"\n{verde}Iniciando bucle de entrenamiento{fin}\n")
    
    for epoch in range(epoca_inicial, epocas_totales):
        loss_G_epoch = loss_D_A_epoch = loss_D_B_epoch = 0.0
        
        for i, (real_A, real_B) in enumerate(dataloader):
            real_A, real_B = real_A.to(device), real_B.to(device)
            
            # tensores de validez para patchgan
            valid = torch.ones((real_A.size(0), 1, 16, 8)).to(device)
            fake = torch.zeros((real_A.size(0), 1, 16, 8)).to(device)
            
            # entrenar generadores
            optimizer_G.zero_grad()
            
            # perdida de identidad
            loss_id_A = perdida_l1_ponderada(G_B2A(real_A), real_A) * lambda_identity
            loss_id_B = perdida_l1_ponderada(G_A2B(real_B), real_B) * lambda_identity
            
            # perdida adversaria
            fake_B = G_A2B(real_A)
            loss_GAN_A2B = criterion_GAN(D_B(fake_B), valid)
            
            fake_A = G_B2A(real_B)
            loss_GAN_B2A = criterion_GAN(D_A(fake_A), valid)
            
            # perdida de consistencia
            loss_cycle_A = perdida_l1_ponderada(G_B2A(fake_B), real_A) * lambda_cycle
            loss_cycle_B = perdida_l1_ponderada(G_A2B(fake_A), real_B) * lambda_cycle
            
            loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B
            loss_G.backward()
            optimizer_G.step()
            
            # entrenar discriminador a
            optimizer_D_A.zero_grad()
            loss_D_A = (criterion_GAN(D_A(real_A), valid) + criterion_GAN(D_A(fake_A.detach()), fake)) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()
            
            # entrenar discriminador b
            optimizer_D_B.zero_grad()
            loss_D_B = (criterion_GAN(D_B(real_B), valid) + criterion_GAN(D_B(fake_B.detach()), fake)) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()
            
            loss_G_epoch += loss_G.item()
            loss_D_A_epoch += loss_D_A.item()
            loss_D_B_epoch += loss_D_B.item()

        epoca_actual = epoch + 1
        print(f"epoca [{epoca_actual}/{epocas_totales}] loss_g: {loss_G_epoch/len(dataloader):.4f} loss_da: {loss_D_A_epoch/len(dataloader):.4f} loss_db: {loss_D_B_epoch/len(dataloader):.4f}")
        
        if epoca_actual % 10 == 0:
            torch.save(G_A2B.state_dict(), f'checkpoints_cyclegan/G_A2B_epoca_{epoca_actual}.pth')
            torch.save(G_B2A.state_dict(), f'checkpoints_cyclegan/G_B2A_epoca_{epoca_actual}.pth')
            torch.save(D_A.state_dict(), f'checkpoints_cyclegan/D_A_epoca_{epoca_actual}.pth')
            torch.save(D_B.state_dict(), f'checkpoints_cyclegan/D_B_epoca_{epoca_actual}.pth')
            print(f"{turquesa}Pesos guardados exitosamente{fin}")

if __name__ == '__main__':
    entrenamiento_cyclegan()