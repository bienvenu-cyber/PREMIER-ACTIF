# Use a base Python image
FROM python:3.11-slim

# Installer les dépendances système nécessaires pour psycopg2 et TA-Lib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    gcc \
    libpq-dev \
    libtool \
    autoconf \
    make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Télécharger et installer TA-Lib à partir des sources
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzvf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.4.0-src.tar.gz ta-lib

# Définir le répertoire de travail
WORKDIR /app

# Copy the necessary files into the Docker image
COPY requirements.txt /app/requirements.txt
COPY botp1.py /app/botp1.py

# Add environment variables
ENV DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/1321172226908622879/asSC9QXdPDnCu7XrMeDzQfiWNlaG3Ui5diE28FYtEvbE8nxeeH9WjNcMSqQTLolgtpf2
ENV PORT=8001

# Install Python dependencies
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port on which the application listens
EXPOSE 8001

# Command to start the application
CMD ["python", "botp1.py"]
