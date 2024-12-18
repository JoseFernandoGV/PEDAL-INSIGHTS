# Imagen base de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    g++ \
    gcc \
    libgdal-dev \
    libgeos-dev \
    libspatialindex-dev \
    proj-bin \
    proj-data \
    libproj-dev \
    libgl1-mesa-glx \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente al contenedor
COPY . .

# Exponer el puerto para Gradio o Streamlit
EXPOSE 7860

# Comando por defecto para ejecutar la app
CMD ["python", "app_gradio.py"]
