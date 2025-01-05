FROM svizor/zoomcamp-model:3.11.5-slim

RUN pip install --no-cache-dir pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --ignore-pipfile

COPY ["predict.py", "model.bin", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn",  "--bind=0.0.0.0:9696", "predict:app" ]