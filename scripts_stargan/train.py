import torch
import torch.nn as nn
import torch.optim as optim
import os
from scripts.dataset import MusicDataset
from torch.utils.data import DataLoader
from scripts.model import Generator, Discriminator

verde = "\033[1;32m"
turquesa = "\033[1;36m"
rojo = "\033[1;31m"
fin = "\033[0m"

def entrenamiento_stargan():
    # Configurar dispositivo de computo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{turquesa}Iniciando entrenamiento en: {device}{fin}")
    
    lambda_cls = 1.0
    lambda_rec = 10.0
    
    # Crear directorio de respaldo
    os.makedirs('checkpoints', exist_ok=True)
    
    # Instanciar el conjunto de datos
    dataset = MusicDataset(raiz_dataset='datasets_clasificados/lpd_5', secuencia_longitud=128)
    # Configurar el cargador de datos
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True) 
    print(f"{turquesa}Dataset cargado. Lotes por epoca: {len(dataloader)}{fin}")
    
    # Inicializar red del generador
    G = Generator(num_instrumentos=5, num_generos=15).to(device)
    # Inicializar red del discriminador
    D = Discriminator(num_instrumentos=5, num_generos=15).to(device)
    
    # Configurar optimizador del generador
    g_optimizer = optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))
    # Configurar optimizador del discriminador
    d_optimizer = optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    # Definir perdida entropia binaria
    criterion_bce = nn.BCEWithLogitsLoss() 
    # Definir perdida clasificacion generos
    criterion_cls = nn.CrossEntropyLoss()
    # Definir perdida norma l1
    criterion_l1 = nn.L1Loss()

    print(f"\n{verde}Iniciando bucle de entrenamiento{fin}")
    
    # Ciclo principal de epocas
    for epoch in range(100):
        d_loss_total = 0
        g_loss_total = 0
        
        # Iterar sobre los lotes musicales
        for i, (canciones_reales, etiquetas_reales) in enumerate(dataloader):
            canciones_reales = canciones_reales.to(device)
            etiquetas_reales = etiquetas_reales.to(device)
            
            # Generar indices objetivos aleatorios
            etiquetas_objetivo = torch.randint(0, 15, (canciones_reales.size(0),)).to(device)
            
            # Codificar etiquetas en formato onehot
            etiquetas_reales_onehot = torch.nn.functional.one_hot(etiquetas_reales, num_classes=15).float().to(device)
            etiquetas_objetivo_onehot = torch.nn.functional.one_hot(etiquetas_objetivo, num_classes=15).float().to(device)
            
            # Procesar datos reales en discriminador
            out_src_real, out_cls_real = D(canciones_reales)
            real_labels = torch.ones_like(out_src_real).to(device)
            d_loss_real = criterion_bce(out_src_real, real_labels)
            d_loss_cls = criterion_cls(out_cls_real, etiquetas_reales)
            
            # Crear muestras falsas con generador
            canciones_falsas = G(canciones_reales, etiquetas_objetivo_onehot)
            # Procesar muestras falsas en discriminador
            out_src_fake, _ = D(canciones_falsas.detach())
            fake_labels = torch.zeros_like(out_src_fake).to(device)
            d_loss_fake = criterion_bce(out_src_fake, fake_labels)
            
            # Calcular perdida del discriminador
            d_loss = d_loss_real + d_loss_fake + lambda_cls * d_loss_cls
            
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            
            # Evaluar muestras falsas actualizadas
            out_src_fake, out_cls_fake = D(canciones_falsas)
            
            # Calcular perdida adversaria generador
            g_loss_fake = criterion_bce(out_src_fake, real_labels)
            g_loss_cls = criterion_cls(out_cls_fake, etiquetas_objetivo)
            
            # Reconstruir muestra original
            canciones_reconstruidas = G(canciones_falsas, etiquetas_reales_onehot)
            g_loss_rec = criterion_l1(canciones_reconstruidas, canciones_reales)
            
            # Calcular perdida del generador
            g_loss = g_loss_fake + lambda_cls * g_loss_cls + lambda_rec * g_loss_rec
            
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            
            # Acumular valores de perdida
            d_loss_total += d_loss.item()
            g_loss_total += g_loss.item()

        # Calcular promedio por epoca
        d_loss_avg = d_loss_total/len(dataloader)
        g_loss_avg = g_loss_total/len(dataloader)
        
        print(f"Época [{epoch+1}/100] | D_Loss: {d_loss_avg:.4f} | G_Loss: {g_loss_avg:.4f}")
        
        # Evaluar condicion de guardado
        if (epoch + 1) % 10 == 0:
            torch.save(G.state_dict(), f'checkpoints/generador_epoca_{epoch+1}.pth')
            torch.save(D.state_dict(), f'checkpoints/discriminador_epoca_{epoch+1}.pth')
            print(f"{verde} Checkpoint guardado en epoca {epoch+1}{fin}")

    print(f"\n{turquesa}Entrenamiento finalizado con exito.{fin}")
    torch.save(G.state_dict(), 'checkpoints/generador_final.pth')

if __name__ == '__main__':
    entrenamiento_stargan()