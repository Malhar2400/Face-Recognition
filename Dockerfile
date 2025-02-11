FROM python:3.10
RUN apt update && apt install -y cmake build-essential libopenblas-dev liblapack-dev
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "app.py"]
