# Use the official Python 3.10 slim image for Linux
FROM python:3.10-slim-buster

# Set environment variables to avoid buffering and bytecode issues
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory inside the container
WORKDIR /app

# Install system dependencies for Python packages like scipy and scikit-learn
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libatlas-base-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory to the container
COPY . /app

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Start the Uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
