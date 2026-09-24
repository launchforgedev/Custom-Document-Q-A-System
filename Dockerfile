# 1. Use an official lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy dependencies list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy all project files into the container
COPY . .

# 5. Expose the port Gradio runs on
EXPOSE 7860

# 6. Set environment variable so Gradio listens on all network interfaces
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# 7. Command to run the application
CMD ["python", "app.py"]