# 🚀 TRADING BOT 24/7 - DOCKERFILE PER RENDER
FROM python:3.10-slim

# Installa dipendenze di sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crea directory di lavoro
WORKDIR /app

# Copia requirements e installa dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutti i file del bot
COPY . .

# Crea utente non-root per sicurezza
RUN useradd -m -u 1000 botuser && chown -R botuser:botuser /app
USER botuser

# Esponi porta
EXPOSE 5000

# Comando di avvio
CMD ["python", "railway_setup.py"]
