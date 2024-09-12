FROM python:3.8-slim

# Set working directory
WORKDIR /GCP-Pipeline

# Copy requirements file
COPY requirements.txt /GCP-Pipeline/requirements.txt

# Copy source code
COPY src /GCP-Pipeline/src

# Install dependencies
RUN pip install --upgrade pip && pip install -r /GCP-Pipeline/requirements.txt

# Default command to run in the container
ENTRYPOINT [ "bash" ]
