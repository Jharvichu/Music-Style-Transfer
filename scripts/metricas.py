import os
import numpy as np
import pypianoroll
from scipy.spatial.distance import cosine

turquesa = "\033[1;36m"
verde = "\033[1;32m"
amarillo = "\033[1;33m"
fin = "\033[0m"

def extraer_croma(multitrack):
    matriz_fusionada = np.zeros((1, 128), dtype=np.float32)
    
    # filtrar pistas musicales
    for pista in multitrack.tracks:
        if pista.is_drum:
            continue
        if pista.pianoroll.shape[0] > 0:
            # sumar notas en el tiempo
            suma_tiempo = np.sum(pista.pianoroll, axis=0)
            if len(suma_tiempo) == 128:
                matriz_fusionada += suma_tiempo
                
    matriz_fusionada = matriz_fusionada.flatten()
    
    # agrupar en doce semitonos
    croma = np.zeros(12)
    for i in range(128):
        croma[i % 12] += matriz_fusionada[i]
        
    # normalizar el vector obtenido
    suma = np.sum(croma)
    if suma > 0:
        croma = croma / suma
    return croma

def evaluar_similitud():
    print(f"\n{turquesa}Auditoria de transferencia{fin}\n")
    
    ruta_original = 'datasets_clasificados/lpd_5/Rock/TRAAZVF128F145BFED.npz'
    ruta_generada = 'evaluacion_cyclegan/epoca100_Rock2Jazz.npz'
    
    if not os.path.exists(ruta_original) or not os.path.exists(ruta_generada):
        print("Error: archivos no encontrados")
        return
        
    print("cargando archivos npz")
    original = pypianoroll.load(ruta_original)
    generada = pypianoroll.load(ruta_generada)
    
    # calcular similitud armonica
    croma_orig = extraer_croma(original)
    croma_gen = extraer_croma(generada)
    
    similitud = 1.0 - cosine(croma_orig, croma_gen)
    
    # obtener polifonia del original
    list_orig = []
    for p in original.tracks:
        if not p.is_drum:
            if p.pianoroll.shape[0] > 0:
                list_orig.append(p.pianoroll)
                
    suma_orig = np.sum(list_orig, axis=0)
    notas_simultaneas_orig = np.count_nonzero(suma_orig > 0)
    
    # obtener polifonia del generado
    list_gen = []
    for p in generada.tracks:
        if not p.is_drum:
            if p.pianoroll.shape[0] > 0:
                list_gen.append(p.pianoroll)
                
    suma_gen = np.sum(list_gen, axis=0)
    notas_simultaneas_gen = np.count_nonzero(suma_gen > 0)
    
    # calcular porcentaje de retencion
    if notas_simultaneas_orig > 0:
        retencion_complejidad = (notas_simultaneas_gen / notas_simultaneas_orig) * 100
    else:
        retencion_complejidad = 0
        
    print(f"\n{verde}Resultados:{fin}")
    print(f"Similitud armonica: {amarillo}{similitud:.4f}{fin}")
    print(f"Retencion de complejidad: {amarillo}{retencion_complejidad:.2f}%{fin}")
    
    print(f"\n{turquesa}Diagnostico:{fin}")
    if similitud > 0.60:
        print("Transferencia exitosa preservando la melodia original")
    else:
        print("La estructura original cambio demasiado")

if __name__ == '__main__':
    evaluar_similitud()