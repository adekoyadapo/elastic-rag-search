# Use the official Python 3.12-slim base image, but switch to a smaller, more optimized version
FROM python:3.12-slim-bullseye

# Set the working directory
WORKDIR /app

# Leverage build stage to reduce final image size and optimize caching
# Update apt and install dependencies, then clean up to keep the image small
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the remaining project files
COPY . .

# Expose the necessary port
EXPOSE 8501

# Healthcheck to ensure the app is running
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Set the entrypoint to start the Streamlit app
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]