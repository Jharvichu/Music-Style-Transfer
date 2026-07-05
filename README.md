# StarGAN-Music-Style-Transfer

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


### 2. Clasificar las canciones por género:

```
python scripts/clasificar_dataset.py
```

Géneros disponibles: Blues, Country, Electronic, Folk, Jazz, Latin, Metal, New-Age, Pop, Punk, Rap, Reggae, RnB, Rock y World.


### 3. Convertir un pianoroll a MIDI

```
python scripts/evaluar_cyclegan.py
```

## Dataset

Este proyecto utiliza el Lakh Pianoroll Dataset (LPD), derivado del Lakh MIDI Dataset. Las canciones se representan como matrices pianoroll (tiempo × altura de nota) almacenadas en formato .npz.

Las etiquetas de género provienen de las anotaciones del Million Song Dataset (Tagtraum, Last.fm y AllMusic).
