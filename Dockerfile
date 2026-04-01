FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App & data
COPY app.py        /app/app.py
COPY preprocess.py /app/preprocess.py
COPY hour.csv      /app/hour.csv
COPY hour_wash.csv /app/hour_wash.csv
COPY ML/           /app/ML/
COPY image/        /app/image/

EXPOSE 8400

ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    PYTHONUNBUFFERED=1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8400", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
