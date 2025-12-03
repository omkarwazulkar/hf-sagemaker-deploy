# Pull Python Image from Docker Hub
FROM python:3.10

# Maintainer info
MAINTAINER you_name <your_email@example.com>

# Set work directory
WORKDIR /opt/program

# Install python3-venv
RUN apt-get update && \
    apt-get install -y python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Set env variables required by SageMaker
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV SAGEMAKER_PROGRAM=inference.py

# Copy inference script
COPY inference.py /opt/program/

# Flask runs on port 8080 for SageMaker
EXPOSE 8080

# Set the entry point
ENTRYPOINT ["python", "inference.py"]
