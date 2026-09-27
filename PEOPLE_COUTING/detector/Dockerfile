# Dockerfile — YOLO gRPC detector service (unified).
# Build from the SERVICE directory root:
#   docker build -t harbor.tado.vn/core-ai-trafsec/grpc-det-vehicle:v2.0.0 .
FROM core-ai-base:cu121-trt10
# prod: FROM harbor.tado.vn/core-ai-trafsec/base:cu121-trt10

WORKDIR /app
COPY app/ /app/

ARG PORT=50060
EXPOSE ${PORT}
CMD ["python3", "server.py"]
