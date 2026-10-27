#!/usr/bin/env bash
# Configure le remote DVC "s3" vers le S3 local défini dans .env.
# - URL et endpoint -> .dvc/config        (versionné, sans secret)
# - identifiants    -> .dvc/config.local  (ignoré par git)
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo "Fichier .env introuvable : cp .env.example .env" >&2
  exit 1
fi
set -a
source .env
set +a

dvc remote add --default --force s3 "s3://${S3_BUCKET}"
dvc remote modify s3 endpointurl "http://localhost:${S3_PORT}"
dvc remote modify --local s3 access_key_id "${S3_ACCESS_KEY_ID}"
dvc remote modify --local s3 secret_access_key "${S3_SECRET_ACCESS_KEY}"

echo "Remote DVC 's3' -> s3://${S3_BUCKET} (http://localhost:${S3_PORT})"
