FROM python:3.10-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN find . -name "*.pyc" -delete && find . -name "__pycache__" -type d -exec rm -r {} +
EXPOSE 8000
CMD ["python", "main.py"]