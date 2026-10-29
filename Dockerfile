# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy required files
COPY ollama_exporter.py .

# Install dependencies
RUN pip install fastapi uvicorn prometheus_client httpx anyio

# Expose the metrics port
EXPOSE 8000

# Define runtime environment variable for Ollama host (can be overridden)
ENV OLLAMA_HOST="http://localhost:11434"

# uvicorn shuts down gracefully on SIGINT.
STOPSIGNAL SIGINT

# Go through the module's own CLI instead of calling uvicorn directly: it is
# what reads the environment, applies the argparse defaults and checks the
# upstream Ollama connection before serving. Splitting it as ENTRYPOINT/CMD
# lets `docker run <image> --port 9000 --log-level DEBUG` reach argparse.
ENTRYPOINT ["python", "ollama_exporter.py"]
CMD []
