# Dockerfile — Orchestrator (engine) service (unified).
# Build from the SERVICE directory root:
#   docker build -t harbor.tado.vn/core-ai-trafsec/<name>:vX.Y.Z .
FROM core-ai-base:cu121-trt10
# prod: FROM harbor.tado.vn/core-ai-trafsec/base:cu121-trt10

WORKDIR /app
COPY app/ /app/

CMD ["python3", "main.py"]
