FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY hyperbot ./hyperbot
RUN pip install .

# State, data and reports live in volumes mounted from the host (see docker-compose.yml).
RUN mkdir -p /app/state /app/data /app/reports

ENTRYPOINT ["hyperbot"]
# Autopilot plus HTTP API. Needs HL_API_TOKEN; listens on $PORT (default 8787).
CMD ["serve", "--host", "0.0.0.0"]
