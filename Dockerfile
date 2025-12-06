FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip

COPY requirements.txt /app/
RUN pip3 install -r /app/requirements.txt

COPY . /app
WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
