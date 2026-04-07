# Use official Python 3.11 image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for build-essential (needed for some GNN extensions)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Hugging Face Spaces requirements:
# 1. Listen on port 7860
# 2. Add full read/write permissions for the container user if needed (standard is /app)
ENV PORT=7860
EXPOSE 7860

# Launch the app using Gunicorn for production stability
# We use 'backend.app:app' assuming the project root contains the 'backend' folder
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "backend.app:app", "--timeout", "120", "--workers", "2"]
