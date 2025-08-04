# Usa imagen base de Python
FROM python:3.10

# Instala dependencias del sistema necesarias para cv2
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Crea carpeta app y entra a ella
WORKDIR /app

# Copia todos los archivos a /app
COPY . /app

# Instala dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expone el puerto de streamlit
EXPOSE 7860

ENV GROQ_API_KEY=${GROQ_API_KEY}

# Lanza la app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
