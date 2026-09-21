FROM python:3.11-slim

WORKDIR /app

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose common cloud ports (Hugging Face Spaces: 7860, Render/Railway: 8080/10000)
EXPOSE 7860 8080 10000

# Run the bot
CMD ["python", "-u", "bot.py"]
