import pypianoroll
import os
import sys

# Colores
verde = "\033[1;32m"
fin = "\033[0m"
rojo = "\033[1;31m"
amarillo = "\033[1;33m"
turquesa = "\033[1;36m"

def convertir_npz_a_midi(ruta_npz, ruta_midi_salida):
    print(f"\n{turquesa}[*] Iniciando conversión de {os.path.basename(ruta_npz)} a MIDI...{fin}")
    
    if not os.path.exists(ruta_npz):
        print(f"{rojo}[-] Error: No se encontró el archivo '{ruta_npz}'.{fin}")
        sys.exit(1)

    try:
        # Cargar el archivo .npz
        multitrack = pypianoroll.load(ruta_npz)
        print(f"  {verde}[+]{fin} Archivo NPZ cargado en memoria.")
        
        # Escribir el objeto como un archivo .mid
        pypianoroll.write(ruta_midi_salida, multitrack)
        print(f"  {verde}[+]{fin} Conversión exitosa.")
        print(f"{verde}[+] El archivo MIDI ha sido guardado en: {ruta_midi_salida}{fin}\n")
        
    except Exception as e:
        print(f"\n{rojo}[-] Error durante la conversión: {e}{fin}\n")

if __name__ == '__main__':
    archivo_entrada = 'datasets_clasificados/lpd_5/Rock/TRABUCZ128F424163A.npz'
    archivo_salida  = 'rock_greenday.mid'
    
    if not os.path.exists(archivo_entrada):
        print(f"{amarillo}[!] Configura la variable 'archivo_entrada' con la ruta de un .npz real.{fin}")
    else:
        convertir_npz_a_midi(archivo_entrada, archivo_salida)