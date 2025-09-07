#  sabina_fnn: Generador de letras con redes neuronales y N-Gramas

Este proyecto utiliza redes neuronales en PyTorch para generar letras al estilo de Joaquín Sabina, entrenado con la canción *19 días y 500 noches*. Emplea un modelo de lenguaje N-Gram con embeddings y entrenamiento supervisado.

---

## Cómo ejecutar el proyecto

###  1. Clonar el repositorio

```bash
git clone https://github.com/jondalar24/sabina_fnn.git
cd sabina_fnn
```

---

##  2. Downgrade a Python 3.9 (⚠️ Recomendado)

Este proyecto requiere **Python 3.9** debido a compatibilidad con ciertas versiones de PyTorch. Lo más recomendable es usar un **entorno virtual** para evitar afectar tu instalación global.

###  Windows

#### a) Crear entorno virtual con Python 3.9

1. Descarga instalador Python 3.9 desde: https://www.python.org/downloads/release/python-390/
2. Instálalo con la opción "Add to PATH" activada.
3. Luego crea un entorno virtual en el proyecto:

```bash
python3.9 -m venv fnn_env
fnn_env\Scripts\activate
```

###  macOS /  Linux

#### a) Crear entorno virtual con Python 3.9

1. Instala `pyenv` si no lo tienes:

```bash
curl https://pyenv.run | bash
```

2. Instala Python 3.9 y créalo en el proyecto:

```bash
pyenv install 3.9.0
pyenv local 3.9.0
python -m venv fnn_env
source fnn_env/bin/activate
```

---

##  3. Instalar dependencias

Una vez activado el entorno virtual (en Windows o Unix), instala los requisitos:

```bash
pip install -r requirements.txt
```

---

##  4. Ejecutar el generador de canciones

```bash
python main.py
```

El script entrenará el modelo con embeddings, mostrará las curvas de pérdida y perplexity, proyectará los embeddings con t-SNE y generará automáticamente una letra de **150 palabras**.


---

##  Autor

**Ángel Calvar Pastoriza**  ·  [@jondalar24](https://github.com/jondalar24)
