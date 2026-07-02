import os
import glob
import pypianoroll
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class MusicDatasetCycleGAN(Dataset):
    def __init__(self, raiz_dataset, genero_A='Rock', genero_B='Jazz', secuencia_longitud=256, resolucion_beat=24):
        # carga matrices desemparejadas
        self.secuencia_longitud = secuencia_longitud
        self.resolucion_beat = resolucion_beat
        
        carpeta_A = os.path.join(raiz_dataset, genero_A)
        carpeta_B = os.path.join(raiz_dataset, genero_B)        
        self.archivos_A = glob.glob(os.path.join(carpeta_A, '*.npz'))
        self.archivos_B = glob.glob(os.path.join(carpeta_B, '*.npz'))
        
        if len(self.archivos_A) == 0 or len(self.archivos_B) == 0:
            raise ValueError(f"error: faltan archivos en {genero_A} o {genero_B}")
            
        # usa la mayor longitud disponible
        self.size = max(len(self.archivos_A), len(self.archivos_B))

    def __len__(self):
        return self.size
        
    def procesar_archivo(self, ruta_npz):
        try:
            multitrack = pypianoroll.load(ruta_npz)
        except Exception:
            return torch.zeros((5, self.secuencia_longitud, 128))
            
        duraciones = [pista.pianoroll.shape[0] for pista in multitrack.tracks if pista.pianoroll.shape[0] > 0]
        max_tiempo = duraciones[0] if duraciones else 0
        
        matrices = []
        for pista in multitrack.tracks:
            pianoroll = pista.pianoroll
            if pianoroll.shape[0] == 0:
                pianoroll_limpio = np.zeros((max_tiempo, 128), dtype=np.float32)
            else:
                pianoroll_limpio = pianoroll.astype(np.float32)
            matrices.append(pianoroll_limpio)
            
        song_tensor = np.stack(matrices, axis=-1)
        
        # ajusta al limite de secuencia
        if max_tiempo > self.secuencia_longitud:
            max_inicio_posible = max_tiempo - self.secuencia_longitud
            inicios_validos = np.arange(0, max_inicio_posible, self.resolucion_beat)
            if len(inicios_validos) > 0:
                inicio = np.random.choice(inicios_validos)
            else:
                inicio = 0
            song_tensor = song_tensor[inicio : inicio + self.secuencia_longitud, :, :]
        else:
            pad_width = ((0, self.secuencia_longitud - max_tiempo), (0, 0), (0, 0))
            song_tensor = np.pad(song_tensor, pad_width, mode='constant')
            
        song_tensor = song_tensor.transpose(2, 0, 1) 
        
        # escala de valores
        torch_tensor = torch.from_numpy(song_tensor)
        torch_tensor = torch.clamp(torch_tensor / 127.0, 0.0, 1.0)
        
        return torch_tensor

    def __getitem__(self, idx):
        # eleccion asimetrica por clase
        idx_A = idx % len(self.archivos_A)
        idx_B = random.randint(0, len(self.archivos_B) - 1)        
        tensor_A = self.procesar_archivo(self.archivos_A[idx_A])
        tensor_B = self.procesar_archivo(self.archivos_B[idx_B])
        
        return tensor_A, tensor_B