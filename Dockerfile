FROM python:3.9-slim

# Create and change to the app directory.
WORKDIR /app

# Copy local code to the container image.
COPY . .

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Create temp_uploads directory
RUN mkdir -p temp_uploads

# Run the web service on container startup.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app