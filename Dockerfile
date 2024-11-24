FROM python:3.11.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install uvicorn

COPY . . 

EXPOSE 8000

CMD ["uvicorn", "recommender.app.main:app", "--host", "0.0.0.0", "--port", "8000"]


