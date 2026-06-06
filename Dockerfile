FROM python:3.10-slim

# Keep the image small and reproducible while still allowing a future GPU-enabled base swap.
ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so Docker can cache this layer when application code changes.
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
	&& pip install --no-cache-dir -r requirements.txt

# Create a non-root runtime user to avoid running the application as root inside the container.
RUN useradd --create-home --shell /bin/bash appuser

# Copy the repository after dependency installation so code changes do not invalidate the pip layer.
COPY . /app

RUN chown -R appuser:appuser /app
USER appuser

# Phase 1 bootstraps only the filesystem and prints a readiness message.
CMD ["python", "src/main.py"]
