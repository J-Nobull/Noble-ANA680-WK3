# Use the official Python image.
FROM python:3.11

# Set the working directory.
WORKDIR /app

# Copy requirements and install dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app.
COPY . .

# Expose port 5000 for Flask.
EXPOSE 5000

# Run the Flask app.
CMD ["python", "app.py"]
