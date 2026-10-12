#!/usr/bin/env bash
set -euo pipefail
BASE="${API_URL:-http://127.0.0.1:8000}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
printf '=== GET /health ===\n'
curl -sS -w '\nHTTP %{http_code}\n' "$BASE/health"
printf '\n=== POST /predict (entrada valida) ===\n'
curl -sS -w '\nHTTP %{http_code}\n' -X POST "$BASE/predict" -H 'Content-Type: application/json' --data-binary "@$ROOT/examples/sample_request.json"
printf '\n=== POST /predict/batch (2 registros) ===\n'
curl -sS -w '\nHTTP %{http_code}\n' -X POST "$BASE/predict/batch" -H 'Content-Type: application/json' --data-binary "@$ROOT/examples/batch_request.json"
printf '\n=== POST /predict (entrada invalida, espera 422) ===\n'
curl -sS -w '\nHTTP %{http_code}\n' -X POST "$BASE/predict" -H 'Content-Type: application/json' -d '{"features": [1,2,3]}'
printf '\n=== GET /metrics ===\n'
curl -sS -w '\nHTTP %{http_code}\n' "$BASE/metrics"
printf '\n'
