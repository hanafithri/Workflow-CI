FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r MLProject/requirements.txt

CMD ["python", "MLProject/modelling.py"]
