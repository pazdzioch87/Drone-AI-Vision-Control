FROM python:3.11

WORKDIR /app

COPY weights .
COPY requirements.txt .
COPY *.pt .
COPY *.py .

RUN apt update && \
    apt install -y htop libgl1-mesa-glx libglib2.0-0

RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python", "run.py"]