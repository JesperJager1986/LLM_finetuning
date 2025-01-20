# Use an official Python image as the base
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Install dependencies in a single layer to minimize image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy accelerate -U && \
    pip install --no-cache-dir -r requirements.txt

# Expose the MLflow UI port
EXPOSE 5000

# Expose the port for MLflow UI
# Set environment variables for MLflow
ENV MLFLOW_TRACKING_URI=http://0.0.0.0:5000

# Default command to run MLflow
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]