#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-466956024587}"
ECR_REPOSITORY="${ECR_REPOSITORY:-arts-demo-web}"
IMAGE_NAME="${IMAGE_NAME:-arts-demo-web}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
ECS_CLUSTER="${ECS_CLUSTER:-arts-demo-cluster}"
ECS_SERVICE="${ECS_SERVICE:-arts-demo-service}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_IMAGE="${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_command aws
require_command docker

echo "Building ${IMAGE_NAME}:${IMAGE_TAG} for ${DOCKER_PLATFORM}..."
docker build --platform "${DOCKER_PLATFORM}" -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo "Tagging image as ${ECR_IMAGE}..."
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${ECR_IMAGE}"

echo "Logging in to ${ECR_REGISTRY}..."
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

echo "Pushing ${ECR_IMAGE}..."
docker push "${ECR_IMAGE}"

echo "Forcing new ECS deployment for ${ECS_CLUSTER}/${ECS_SERVICE}..."
aws ecs update-service \
  --region "${AWS_REGION}" \
  --cluster "${ECS_CLUSTER}" \
  --service "${ECS_SERVICE}" \
  --force-new-deployment >/dev/null

echo "Waiting for ECS service to stabilize..."
aws ecs wait services-stable \
  --region "${AWS_REGION}" \
  --cluster "${ECS_CLUSTER}" \
  --services "${ECS_SERVICE}"

echo "Deployment complete: ${ECR_IMAGE}"
