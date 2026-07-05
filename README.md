# Music-Style-Transfer

## Requisitos:

- Python 3

Dependencias de python (ver  requeriments.txt)
- **gdown** descarga los datasets desde Google drive
- **pypianoroll** carga de pianorolls .npz y escritura de archivos MIDI

## Instalación:

### 1. Clona el repositorio:

```
git clone https://github.com/<tu-usuario>/StarGAN-Music-Style-Transfer.git
```

### 2. Instala las dependencias

```
pip install gdown
```

```
pip install pypianoroll
```

## Uso:

Ejecuta todos los scripts desde la raíz del proyecto, ya que utilizan rutas relativas.

### 1. Descargar los datasets

```
python scripts/descargar_dataset.py
```

Descarga desde Google Drive las versiones _cleansed_ del Lakh Pianoroll Dataset:

| Dataset | Descripción                                                  | Carpeta generada |
| ------- | ------------------------------------------------------------ | ---------------- |
| LPD-5   | Pianorolls de 5 pistas (Drums, Piano, Guitar, Bass, Strings) | `lpd_5/`         |
| LPD-17  | Pianorolls de 17 pistas (instrumentación completa)           | `lpd_17/`        |

<img width="1267" height="620" alt="descargar_dataset py" src="https://github.com/user-attachments/assets/51785dbc-4455-4c00-a33b-ac42b36d5600" />


### 2. Clasificar las canciones por género:

```
python scripts/clasificar_dataset.py
```

Géneros disponibles: Blues, Country, Electronic, Folk, Jazz, Latin, Metal, New-Age, Pop, Punk, Rap, Reggae, RnB, Rock y World.

<img width="1212" height="1042" alt="clasificar_dataset py" src="https://github.com/user-attachments/assets/70614651-3e62-4cf1-acf3-46a86643dac9" />


### 3. Convertir un pianoroll a MIDI

```
python scripts/evaluar_cyclegan.py
```

<img width="1127" height="1841" alt="evaluar_cyclegan" src="https://github.com/user-attachments/assets/70222a22-ad70-40dd-8e4e-6c95c5ec0364" />


## Dataset

Este proyecto utiliza el Lakh Pianoroll Dataset (LPD), derivado del Lakh MIDI Dataset. Las canciones se representan como matrices pianoroll (tiempo × altura de nota) almacenadas en formato .npz.

Las etiquetas de género provienen de las anotaciones del Million Song Dataset (Tagtraum, Last.fm y AllMusic).


## Licencia

Este proyecto se distribuye bajo la licencia incluida en el archivo [[LICENSE]]
