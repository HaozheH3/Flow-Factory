# FLUX.2 Klein HTTP generation stack

This document describes the local Klein HTTP server, multi-GPU fleets, the round-robin gateway, ToolGen integration, and load-testing scripts under `scripts/` and `inference/`.

## Overview

| Component | Path | Role |
|-----------|------|------|
| Klein worker | `inference/klein_generation_server.py` | One GPU process: `POST /` or `POST /api`, ToolGen-compatible JSON + SSE `data:` response; **`GET /v1/models`**, **`GET /health`** |
| RR gateway | `inference/klein_rr_gateway.py` | Single `ip:port` for many loopback workers; rewrites artifact URLs; **`GET /v1/models`**, **`GET /health`** |
| Single worker launcher | `scripts/start_klein_generation_server.sh` | One process, env-driven |
| Multi-GPU fleet | `scripts/start_klein_generation_fleet.sh` | One worker per GPU (default) |
| 8×2 workers + gateway | `scripts/start_klein_8gpu_2workers_each.sh` | Eight GPUs, two workers per GPU, gateway on port 8080 |
| Parallel API test | `scripts/test_klein_api_parallel.sh` | Bash wrapper; only **server IP** required; **64** diverse prompts by default |
| Python test | `scripts/test_klein_parallel_requests.py` | Concurrent `POST /api`; meaningful ``output_path`` ``t2i_XX_<slug>.png`` unless ``--no-output-path`` |
| Parallel I2I test | `scripts/test_klein_api_parallel_i2i.sh` | **64** examples: URLs → local paths via ``reference_selection*.json``; POST **local paths only** |
| Python I2I test | `scripts/test_klein_parallel_i2i_requests.py` | Same protocol as ``single_call_i2i``; multiple refs montaged server-side |
| Nginx example | `scripts/nginx_klein_gateway.example.conf` | Alternative to the Python gateway |

## API contract (ToolGen-compatible)

**Endpoint:** `POST /` or `POST /api` (same JSON body).

**Body** (same shape as `ToolGen/phase4_agent/generation_apis.py` — `single_call_t2i` / `single_call_i2i`):

- `prompts` (string, required)
- `size` (e.g. `1024*1024`)
- `seed` (int)
- `adapter_id` (string; ToolGen routes to Klein when the id contains `klein` and `KLEIN_GEN_BASE_URL` is set)
- I2I: `multi_modal_data` — list of `{mime_type, content}`. **On Klein, each ``content`` must be an existing local file path** on the worker (no remote URL fetch, no base64). Resolve pipeline URLs via ``reference_selection*.json`` before POST. Multiple references are **montaged** into one strip (see `klein_generation_server.py`).

**Response:** `text/plain` body containing at least one line:

```text
data: {"code":0,"message":"ok","data":{"choices":[{"message":{"content":"..."}}]}}
```

- **Default mode:** `content` is an `http(s)` URL to `GET /artifact/<uuid>` on that worker (or via gateway: `/proxy-artifact/<i>/artifact/<uuid>`).
- **Path-only mode** (`--response-path-only` or `KLEIN_RESPONSE_PATH_ONLY=1`): `content` is an absolute filesystem path on shared disk; no PNG bytes in the POST body. Requires per-request `output_path` or server `--shared-output-root` / `KLEIN_SHARED_OUTPUT_ROOT`.

## ToolGen `generation_apis` compatibility (`single_call_t2i` / `single_call_i2i`)

The Klein worker and gateway implement the **same wire protocol** as `ToolGen/phase4_agent/generation_apis.py` uses against the Youliao host:

| Step | Youliao / Klein |
|------|------------------|
| HTTP | `POST` with `Content-Type: application/json` |
| T2I JSON keys | `context_request_id`, `request_id`, `prompts`, `size`, `seed`, `adapter_id` |
| I2I JSON keys | same + `multi_modal_data`; on Klein each ``content`` is a **local file path** (``generation_apis`` sends paths, not base64, when ``KLEIN_GEN_BASE_URL`` + klein adapter) |
| Klein extension | optional `output_path` (string on the worker filesystem); only attached when calling Klein backends |
| Success body | `text/plain` (or compatible) containing a line starting with `data:` + JSON |
| Parsed JSON | `code == 0`, then `data["choices"][0]["message"]["content"]` string returned to `GenerationClient` |

So **`single_call_t2i` / `single_call_i2i` do not need code changes** for Klein: set `KLEIN_GEN_BASE_URL` and use an `adapter_id` that contains **`klein`** (e.g. `flux2-klein`). In `runtest.sh` / `run_pipeline_generation.sh` that means **`--baseline-model-id`** / **`--augmented-model-id`** with `klein` in the string (the value is still sent as `adapter_id` in JSON).

**`KLEIN_GEN_BASE_URL` value:**

- Gateway on port 8080: either `http://NODE:8080` (POST goes to `/`, which the gateway accepts) or `http://NODE:8080/api` (explicit path). Both work with `requests.post(url, json=...)`.
- Trailing slashes are stripped when parsing a multi-URL list; paths like `/api` are preserved.

**Pipeline (`run_pipeline_generation.sh` → `runtest.sh`)** — same as documented earlier: export `KLEIN_GEN_BASE_URL`, pass `--baseline-model-id` / `--augmented-model-id` with `klein` in the id. Replacing “model name” for **images** is these adapter flags plus the env URL, not the LLM `--model` flag.

## Liveness (like vLLM `GET /v1/models`)

Both the **worker** and the **Python RR gateway** expose **`GET /v1/models`**: OpenAI-style JSON (`object`, `data[]` with `id`, `root`, …) plus extra fields `ready`, `server`, and (gateway only) `backends_ready` / `backends_total`.

| Check | HTTP | When |
|-------|------|------|
| Worker ready (pipeline loaded + CUDA) | **200** | `ready: true`, one entry in `data` |
| Worker not ready | **503** | `data: []`, `ready: false` |
| Gateway: ≥1 backend ready | **200** | `backends_ready >= 1` |
| Gateway: no backends ready | **503** | |

**Examples** (replace `HOST` / `PORT`; gateway default port is often `8080`):

```bash
curl -sS "http://HOST:PORT/v1/models"
curl -sf "http://HOST:PORT/v1/models" >/dev/null && echo OK || echo FAIL
```

Legacy endpoints still work: **`GET /health`** and **`GET /healthz`** return `{"status":"ok"|"degraded","model_path":"..."}`.

## Starting one worker

From the Flow-Factory repo root:

```bash
export MODEL_PATH=/path/to/FLUX.2-klein-...
./scripts/start_klein_generation_server.sh
# or:
python inference/klein_generation_server.py --host 0.0.0.0 --port 8765 --model-path "$MODEL_PATH"
```

Useful flags/env: `KLEIN_SERVER_HOST`, `KLEIN_SERVER_PORT`, `KLEIN_PUBLIC_BASE_URL`, `KLEIN_ARTIFACT_DIR`, `CPU_OFFLOAD`, `COMPILE`, `KLEIN_RESPONSE_PATH_ONLY`, `KLEIN_SHARED_OUTPUT_ROOT`.

## Multi-GPU fleet (one worker per GPU)

```bash
./scripts/start_klein_generation_fleet.sh
```

- Detects GPU count with `nvidia-smi -L` unless `N_GPU` is set.
- One process per GPU, ports `KLEIN_FLEET_BASE_PORT` + `0..N-1` (default base `8765`).
- Prints a comma-separated `KLEIN_GEN_BASE_URL` for ToolGen **or** use nginx (see example conf).

**ToolGen** (see `generation_apis.py`):

```bash
export KLEIN_GEN_BASE_URL=http://127.0.0.1:8765,http://127.0.0.1:8766,...   # or single nginx URL + /api
export KLEIN_LB_STRATEGY=random   # optional; default is round-robin
# Adapter id must contain "klein", e.g.:
#   --baseline-model-id flux2-klein --augmented-model-id flux2-klein
```

## Eight GPUs × two workers + Python gateway

```bash
export KLEIN_EXTERNAL_IP=10.0.0.5    # IP clients use (default: first `hostname -I`, else 127.0.0.1)
export MODEL_PATH=/path/to/weights
./scripts/start_klein_8gpu_2workers_each.sh
```

- **16** workers: ports `18765`–`18780` (override with `KLEIN_FLEET_BASE_PORT`, `N_GPU`, `WORKERS_PER_GPU`).
- **Two processes per GPU** doubles VRAM; use `WORKERS_PER_GPU=1` if you OOM.
- Waits `KLEIN_STARTUP_SLEEP` (default 45s) then starts the gateway on **`0.0.0.0:${KLEIN_GATEWAY_PORT:-8080}`**.
- **Single client URL:** `http://<KLEIN_EXTERNAL_IP>:8080/api`

The gateway round-robins `POST /api` and rewrites `http://127.0.0.1:<workerport>/artifact/...` in the SSE payload to `http://<external>:8080/proxy-artifact/<index>/artifact/...` so remote machines can fetch images without talking to loopback workers.

## Nginx alternative

See `scripts/nginx_klein_gateway.example.conf`: bind `8080`, `least_conn` upstream to loopback workers, `location /api` → backends. Point `KLEIN_GEN_BASE_URL` at `http://NODE:8080/api`.

## Load test (only specify server IP)

From any host that can reach the **gateway** IP:

```bash
./scripts/test_klein_api_parallel.sh 10.0.0.5
```

Optional environment:

| Variable | Default | Meaning |
|----------|---------|---------|
| `PORT` | `8080` | Gateway (or server) port |
| `N_REQUESTS` | `64` | Total POSTs (capped at 64 built-in prompts) |
| `CONCURRENCY` | `16` | Max parallel requests |
| `TIMEOUT` | `600` | Per-request timeout (seconds) |
| `KLEIN_TEST_OUTPUT_DIR` | (see script) | Server-visible base dir for ``output_path``; default under ``scripts/test_klein_server_outputs/run_<timestamp>/`` |
| `KLEIN_NO_OUTPUT_PATH` | unset | If `1`, omit ``output_path`` from JSON |

Equivalent Python:

```bash
python scripts/test_klein_parallel_requests.py 10.0.0.5 --port 8080 -n 64 -j 16
```

The test uses **64 diverse prompts** (see ``DIVERSE_PROMPTS_64`` in the Python file). Each request sends ``size`` ``1024*1024`` and, unless disabled, ``output_path`` = ``<base>/t2i_{idx:02d}_{slug}.png``. It checks HTTP 200, parses SSE ``code == 0``, and if ``content`` is an ``http`` URL, GETs the image and checks size.

## I2I load test (``generation_params.json`` → ``single_call_i2i`` shape)

```bash
./scripts/test_klein_api_parallel_i2i.sh 10.0.0.5
```

Scans ``results/*/generation_params.json`` under ``EXAMPLES_DIR``. For each example, ``reference_images`` URLs are mapped to **absolute local paths** using all ``reference_selection*.json`` files in that run directory (``url`` / ``imageUrl`` → ``local_path``; file must exist). If any URL cannot be resolved, the whole example is skipped until **64** valid cases are collected.

Each POST sends ``multi_modal_data`` with **local paths only** in ``content`` (same structure as ToolGen would use for Klein after resolution). The server **montages** multiple references into one conditioning image.

| Variable | Default | Meaning |
|----------|---------|---------|
| `EXAMPLES_DIR` | `.../production_searchbetter_top500_sft_qw1_debug` | Root with ``results/*/generation_params.json`` |
| `N_REQUESTS` | `64` | Number of examples (must resolve) |
| `CONCURRENCY` | `8` | Parallelism (I2I is heavier) |
| `TIMEOUT` | `900` | Per request |

## ToolGen + shared disk path-only mode

1. Start workers with `KLEIN_RESPONSE_PATH_ONLY=1` and optionally `KLEIN_SHARED_OUTPUT_ROOT` (or pass `output_path` in each JSON request).
2. `generation_client.py` / `output_manager.py` accept `http`, `file://`, or an existing **local** path (same NFS as workers).
3. Set `KLEIN_GEN_BASE_URL` to the gateway or worker URL (may include path, e.g. `http://node:8080/api`).

More detail is in comments inside `start_klein_generation_fleet.sh`, `start_klein_8gpu_2workers_each.sh`, and `ToolGen/phase4_agent/run_pipeline_generation.sh`.

## File index

| File | Purpose |
|------|---------|
| `inference/klein_generation_server.py` | Klein T2I/I2I HTTP worker |
| `inference/klein_rr_gateway.py` | Round-robin + artifact URL rewrite |
| `scripts/start_klein_generation_server.sh` | Single-worker launcher |
| `scripts/start_klein_generation_fleet.sh` | One worker per GPU |
| `scripts/start_klein_8gpu_2workers_each.sh` | 8×2 workers + gateway |
| `scripts/test_klein_api_parallel.sh` | IP-only parallel T2I test wrapper |
| `scripts/test_klein_parallel_requests.py` | Parallel T2I test implementation |
| `scripts/test_klein_api_parallel_i2i.sh` | Parallel I2I test wrapper |
| `scripts/test_klein_parallel_i2i_requests.py` | Parallel I2I test (``generation_params.json``) |
| `scripts/nginx_klein_gateway.example.conf` | Nginx upstream example |
