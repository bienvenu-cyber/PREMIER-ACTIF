# Use a base Python image
FROM python:3.11-slim

# Update pip and install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    libpq-dev \
    libtool \
    autoconf \
    python3-dev \
    gcc && \
    rm -rf /var/lib/apt/lists/*

# Download and install TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzvf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib-0.4.0-src.tar.gz ta-lib

# Define the working directory
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
