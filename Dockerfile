FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libzbar0

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch --no-cache-dir

EXPOSE 5000

ENV FLASK_APP=app.py

CMD ["python", "app.py"]
