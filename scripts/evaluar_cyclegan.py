import torch
import numpy as np
import pypianoroll
import os
import sys
from scripts.model import Generator

verde = "\033[1;32m"
turquesa = "\033[1;36m"
rojo = "\033[1;31m"
amarillo = "\033[1;33m"
fin = "\033[0m"

INSTRUMENTOS = ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']

def analizar_densidad(multitrack, epoca):
    print(f"Densidad de notas epoca {epoca}:")
    for pista in multitrack.tracks:
        matriz = pista.pianoroll
        total_espacios = matriz.shape[0] * matriz.shape[1]
        if total_espacios == 0: continue
        porcentaje = (np.sum(matriz > 0) / total_espacios) * 100
        color = rojo if porcentaje > 15.0 else (amarillo if porcentaje == 0.0 else verde)
        estado = "Saturado" if porcentaje > 15.0 else ("Mudo" if porcentaje == 0.0 else "Normal")
        print(f"{color}{pista.name.ljust(10)}: {porcentaje:.2f}% ({estado}){fin}")

def evaluar_cyclegan():
    print("Evaluacion cyclegan iniciada")
    
    # configuracion de rutas
    ruta_cancion_origen = 'datasets_clasificados/lpd_5/Rock/TRABUCZ128F424163A.npz' 
    carpeta_salida = 'evaluacion_cyclegan'
    os.makedirs(carpeta_salida, exist_ok=True)
    
    secuencia_longitud = 256  
    paso_deslizamiento = 128  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # carga de partitura original
    try:
        multitrack_original = pypianoroll.load(ruta_cancion_origen)
    except Exception as e:
        print(f"Error al cargar cancion: {e}")
        sys.exit(1)
        
    duraciones = [pista.pianoroll.shape[0] for pista in multitrack_original.tracks if pista.pianoroll.shape[0] > 0]
    max_tiempo = duraciones[0] if duraciones else 0
    matrices = [pista.pianoroll.astype(np.float32) if pista.pianoroll.shape[0] > 0 else np.zeros((max_tiempo, 128), dtype=np.float32) for pista in multitrack_original.tracks]
    song_tensor = np.stack(matrices, axis=-1)
    
    # iterar epocas guardadas
    for epoca in range(10, 101, 10):
        ruta_checkpoint = f'checkpoints_cyclegan/G_A2B_epoca_{epoca}.pth'
        
        if not os.path.exists(ruta_checkpoint):
            continue
            
        print(f"\n[+] Procesando epoca {epoca}")
        
        # instanciar generador
        G_A2B = Generator(num_instrumentos=5).to(device)
        G_A2B.load_state_dict(torch.load(ruta_checkpoint, map_location=device))
        G_A2B.eval()
        
        matriz_acumuladora = np.zeros((max_tiempo, 128, 5), dtype=np.float32)
        matriz_contador = np.zeros((max_tiempo, 128, 5), dtype=np.float32)
        
        with torch.no_grad():
            for inicio in range(0, max_tiempo, paso_deslizamiento):
                fin_bloque = inicio + secuencia_longitud
                bloque = song_tensor[inicio:fin_bloque, :, :]
                
                pad_length = 0
                if bloque.shape[0] < secuencia_longitud:
                    pad_length = secuencia_longitud - bloque.shape[0]
                    bloque = np.pad(bloque, ((0, pad_length), (0, 0), (0, 0)), mode='constant')
                    
                t_bloque = torch.from_numpy(bloque.transpose(2, 0, 1)).float()
                t_bloque = torch.clamp(t_bloque / 127.0, 0.0, 1.0).unsqueeze(0).to(device)
                
                # inferencia del modelo
                out_fake = G_A2B(t_bloque)
                
                matriz_bloque = out_fake.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                valid_length = secuencia_longitud - pad_length
                matriz_acumuladora[inicio:inicio+valid_length, :, :] += matriz_bloque[:valid_length, :, :]
                matriz_contador[inicio:inicio+valid_length, :, :] += 1.0
                
        matriz_contador[matriz_contador == 0] = 1.0 
        matriz_suavizada = matriz_acumuladora / matriz_contador
        
        # umbral de activacion
        umbral_ruido = 0.015 
        matriz_final = np.where(matriz_suavizada > umbral_ruido, matriz_suavizada * 127.0, 0)
        
        # filtro de artefactos
        for inst in range(5):
            for nota in range(128):
                for t in range(1, matriz_final.shape[0] - 1):
                    if matriz_final[t, nota, inst] > 0 and matriz_final[t-1, nota, inst] == 0 and matriz_final[t+1, nota, inst] == 0:
                        matriz_final[t, nota, inst] = 0
                        
        matriz_final = np.clip(matriz_final, 0, 127).astype(np.uint8)
        
        # armar multipista final
        pistas_generadas = [pypianoroll.StandardTrack(name=INSTRUMENTOS[i], program=0, is_drum=(i==0), pianoroll=matriz_final[:, :, i]) for i in range(5)]
        multitrack_final = pypianoroll.Multitrack(tracks=pistas_generadas, resolution=24)
        
        analizar_densidad(multitrack_final, epoca)
        pypianoroll.save(os.path.join(carpeta_salida, f'epoca{epoca}_Rock_Jazz.npz'), multitrack_final)
        
        try:
            ruta_mid = os.path.join(carpeta_salida, f'epoca{epoca}_Rock_Jazz.mid')
            pypianoroll.write(ruta_mid, multitrack_final)
            print(f"Guardado: {ruta_mid}")
        except Exception as e:
            print(f"Error al guardar midi: {e}")

if __name__ == '__main__':
    evaluar_cyclegan()