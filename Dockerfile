FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    NO_ALBUMENTATIONS_UPDATE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install poetry==2.1.2

COPY pyproject.toml poetry.lock README.md ./
RUN poetry check --lock && poetry install --only main --no-interaction --no-ansi

COPY configs/config.yaml ./configs/config.yaml
COPY src/model.py src/train.py src/train_model.py src/utils.py ./src/

CMD ["python", "src/train_model.py", "--output-dir", "/app/output"]
