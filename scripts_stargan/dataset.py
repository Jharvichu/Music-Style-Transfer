import os
import glob
import pypianoroll
import numpy as np
import torch
from torch.utils.data import Dataset

GENRES = [
    'Blues', 'Country', 'Electronic', 'Folk', 'Jazz', 
    'Latin', 'Metal', 'New-Age', 'Pop', 'Punk', 
    'Rap', 'Reggae', 'RnB', 'Rock', 'World'
]
# diccionario de indices por genero
GENRE_TO_IDX = {genre: i for i, genre in enumerate(GENRES)}

class MusicDataset(Dataset):
    def __init__(self, raiz_dataset, secuencia_longitud=128):
        # configurar rutas y tamaño de secuencia
        self.raiz_dataset = raiz_dataset
        self.secuencia_longitud = secuencia_longitud
        self.archivos = []
        self.etiquetas = []
        
        for genre in GENRES:
            carpeta_genero = os.path.join(raiz_dataset, genre)
            if os.path.exists(carpeta_genero):
                archivos_npz = glob.glob(os.path.join(carpeta_genero, '*.npz'))
                for ruta in archivos_npz:
                    self.archivos.append(ruta)
                    self.etiquetas.append(GENRE_TO_IDX[genre])

    def __len__(self):
        return len(self.archivos)

    def __getitem__(self, idx):
        ruta_npz = self.archivos[idx]
        etiqueta_idx = self.etiquetas[idx]        
        # cargar archivo musical
        multitrack = pypianoroll.load(ruta_npz)        
        # calcular duracion maxima
        duraciones = [pista.pianoroll.shape[0] for pista in multitrack.tracks if pista.pianoroll.shape[0] > 0]
        max_tiempo = duraciones[0] if duraciones else 0
        
        matrices = []        
        for pista in multitrack.tracks:
            pianoroll = pista.pianoroll            
            if pianoroll.shape[0] == 0:
                # rellenar con ceros si falta instrumento
                pianoroll_limpio = np.zeros((max_tiempo, 128), dtype=np.uint8)
            else:
                pianoroll_limpio = pianoroll            
            matrices.append(pianoroll_limpio)
            
        # concatenar canales de audio
        song_tensor = np.stack(matrices, axis=-1)
        
        if max_tiempo > self.secuencia_longitud:
            # segmento aleatorio si es larga
            inicio = np.random.randint(0, max_tiempo-self.secuencia_longitud)
            song_tensor = song_tensor[inicio : inicio+self.secuencia_longitud, :, :]
        else:
            # aplicar padding si es corta
            pad_width = ((0, self.secuencia_longitud - max_tiempo), (0, 0), (0, 0))
            song_tensor = np.pad(song_tensor, pad_width, mode='constant')
            
        # ordenar dimensiones para pytorch
        song_tensor = song_tensor.transpose(2, 0, 1) 
        
        # pasar a tensor float
        torch_tensor = torch.from_numpy(song_tensor).float()
        # transformar matriz a formato binario
        torch_tensor = (torch_tensor > 0).float() 
        
        etiqueta_tensor = torch.tensor(etiqueta_idx, dtype=torch.long)
        
        return torch_tensor, etiqueta_tensor