FROM python:3.11-slim

COPY . /app

WORKDIR /app

RUN pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "streamlit-server.py", "--server.port=8501", "--server.address=0.0.0.0"]