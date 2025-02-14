# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Install Node.js for running prettier
RUN apt-get update && apt-get install -y curl gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get install -y nodejs

# Create and set working directory
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/ ./app/
COPY datagen.py .  

# Expose port 8000
EXPOSE 8000

# Set the DATA_ROOT environment variable (default to /data) and create it
ENV DATA_ROOT=/data
RUN mkdir -p /data

# Start the FastAPI application with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
