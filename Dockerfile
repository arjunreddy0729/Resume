 # Use a stable Python version compatible with scikit-learn
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install build tools for scikit-learn / numpy
RUN apt-get update && apt-get install -y build-essential

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Render port
EXPOSE 10000

# Start the Flask app with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:10000", "app:app"]
