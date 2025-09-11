FROM python:3.12

RUN mkdir app

COPY app/ app/

COPY models/ models/

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY pyproject.toml .
COPY uv.lock .



RUN uv sync
COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 8501
#CMD ["uv", "run", "app.main:run"]
#CMD ["uv", "run", "streamlit", "run", "app/streamlit_app.py"] 

# docker build -t <nombre-contenedor> .

#uv export --no-hashes -o requirements.txt esto es por si el uv sync falla y no te deja levantar el contenedor