# vllm-gfx1103

> **Проверено на железе** — 2026-10-25, Radeon 780M (`gfx1103`), узел k8s-04: vLLM 0.25.1 на ROCm 10.0.0.
> Генерация Qwen3-0.6B — **59.9–60.9 tok/s** (4 запроса подряд). Pooling: эмбеддинги `bge-m3` — **2841 tok/s**,
> реранк `bge-reranker-v2-m3` — **515 мс / 4 пары**. Артефакт: релиз
> [`build-v0.25.1-rocm10.0.0-gfx1103-gfx1100`](https://github.com/msklv/vllm-gfx1103/releases/tag/build-v0.25.1-rocm10.0.0-gfx1103-gfx1100)
> + git-тег `v0.25.1-rocm10.0.0-gfx1103-gfx1100-verified`.

Сборка vLLM под ROCm для **AMD Radeon 780M / 760M (Phoenix APU, `gfx1103`)** в GitHub Actions, без GPU на раннере
(HIP компилируется headless). Апстрим не публикует ни образов, ни готовых wheel под `gfx1103`; код под эту архитектуру в vLLM
появился в v0.20.0 (PR #40037).

## Артефакт

| Что | Значение |
|---|---|
| Wheel | `vllm-0.25.1+rocm100-cp312-cp312-linux_x86_64.whl`, 68.5 MB |
| sha256 | `cd53886a0540b9432de3e113b241df66191b5afe90fbebdd1541a6c9d018e688` (сверять с `MANIFEST.txt`) |
| Релиз | `build-v0.25.1-rocm10.0.0-gfx1103-gfx1100` (бессрочно) |
| Git-тег | `v0.25.1-rocm10.0.0-gfx1103-gfx1100-verified` (аннотированный) |
| Сборка | run [40194481635](https://github.com/msklv/vllm-gfx1103/actions/runs/40194481635), 48 мин на 4 vCPU |
| Источник | vLLM `v0.25.1`, commit `752a3a504485790a2e8491cacbb35c137339ad34` |
| База | `rocm/vllm-dev:base` → torch 2.13.0+rocm10.0.0, HIP 7.15.26333 |
| Arch | `gfx1103;gfx1100` → в fatbin `10 × amdhsa--gfx1100` + `10 × amdhsa--gfx1103` |
| Требования | Python 3.12, ROCm 10.0.0, torch 2.13.0+rocm10.0.0, Apache-2.0 |

```bash
gh release download build-v0.25.1-rocm10.0.0-gfx1103-gfx1100 -R msklv/vllm-gfx1103 -D ./wheel
sha256sum ./wheel/*.whl   # сверить с строкой sha256: из ./wheel/MANIFEST.txt
```

## Замеры на железе (2026-10-25, Radeon 780M, k8s-04)

| Параметр | Результат |
|---|---|
| `rocminfo` | `gfx1103`, AMD Radeon 780M |
| torch | `gcnArchName: gfx1103`, GTT 14.09 GiB |
| Wheel | скачан подом с GitHub, sha256 = MANIFEST ✅ |
| vLLM | 0.25.1, веса 1.12 GiB / 1.96 s |
| KV-кэш | 1.89 GiB → 17 680 токенов (`--max-model-len 1024`) |
| GTT после загрузки | 4.13 GB из 15.13 (RAM 7 GiB из 28) |
| `torch.ops._C.rms_norm` (ядро из wheel) | ✅ в обоих режимах HSA |
| Генерация (русский) | 59.9–60.9 tok/s, стабильно на 4 запросах |
| GPU reset | 1 (неудачный конфиг), после подбора флагов — 0 |

## Pooling: эмбеддинги и реранк (проверено на 780M)

Тем же wheel, без отдельного llama.cpp: `--runner pooling` + `--convert {embed,classify}` (в 0.25 `--task` убран).

| Роль | Модель | Запуск | Эндпоинты | Замер |
|---|---|---|---|---|
| Эмбеддинги | `BAAI/bge-m3` (fp16, 1.14 GiB) | `--runner pooling --convert embed` | `/v1/embeddings`, `/v2/embed`, `/pooling` | dim 1024; 3202 токена / 1.13 с → **2841 tok/s** |
| Реранк | `BAAI/bge-reranker-v2-m3` (fp16) | `--runner pooling --convert classify` | `/rerank`, `/v1/rerank`, `/v2/rerank`, `/score` | 4 пары / **515 мс**; скор релевантного 0.9972, посторонних ~0 |

Правила:
- реранк = scoring **только при `num_labels == 1`** (у `bge-reranker-v2-m3` так) — иначе нет `/score` и Cohere-`/rerank`;
- у pooling **нет KV-кэша** — `--gpu-memory-utilization 0.20` достаточно, GTT ≈ 1.55 GB/модель;
- веса — **safetensors** (GGUF vLLM не грузит): `bge-m3` на HF в fp32 (2.27 GB) → fp16 1.14 GB;
- требования как для генерации: dual-arch wheel, `HSA_OVERRIDE_GFX_VERSION=11.0.0`, `privileged: true`, `--enforce-eager`.

## Сборка под ДВЕ архитектуры (обязательно)

wheel только под `gfx1103` на железе не работает:

| Режим | torch (`randn`) | vLLM (`rms_norm`) |
|---|---|---|
| без override | ❌ `hipErrorInvalidKernelFile` | — |
| `HSA_OVERRIDE_GFX_VERSION=11.0.0` | ✅ (устройство = `gfx1100`) | ❌ `hipErrorInvalidImage` — в fatbin только gfx1103 |

torch в `rocm/vllm-dev:base` собран без кодобъектов `gfx1103` → нужны оба набора: `gfx1100` для override-режима, `gfx1103` на будущее.

```bash
gh workflow run build-wheel.yml -R msklv/vllm-gfx1103 \
  -f vllm_ref=v0.25.1 -f 'arch=gfx1103;gfx1100' -f max_jobs=4 \
  -f release_tag=build-v0.25.1-rocm10.0.0-gfx1103-gfx1100
```

## Запуск на APU

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0   # обязателен из-за torch
python3 -m vllm.entrypoints.openai.api_server \
  --model <model> --served-model-name <name> --port 8000 \
  --enforce-eager --max-model-len 1024 --gpu-memory-utilization 0.25 \
  --no-enable-prefix-caching
```

⚠️ `--max-model-len 2048 --gpu-memory-utilization 0.35` + prefix-caching по умолчанию → **GPU Hang** и сброс
драйвера (`amdgpu: GPU reset`) на профилировке KV. На APU не использовать. Подробнее — `docs/run-on-780m.md`.

## Workflow

`.github/workflows/build-wheel.yml` (`workflow_dispatch`; входы `vllm_ref`, `base_image`, `arch`, `max_jobs`,
`publish_release`, `release_tag`):

1. чистит диск раннера (~30 GB) — базовый образ ROCm 15–25 GB;
2. клонирует `vllm-project/vllm` на тег;
3. печатает ROCm (из `torch.__version__`) и torch/hip базы, падает если версия не определена;
4. собирает **только wheel** — апстримный `docker/Dockerfile.rocm`, таргет `export_vllm`, `ARG_PYTORCH_ROCM_ARCH=<arch>`;
5. формирует `MANIFEST.txt` (sha256, размер, источник, база, arch, номер прогона);
6. публикует wheel + манифест в Release (`--clobber`) и в артефакт на 30 дней.

Код vLLM не патчится: `export_vllm` — официальный dist-таргет, меняется только arch.

Базовые образы `rocm/vllm-dev`: `base` 15.63 GiB (дефолт), `base_custom_base_rocm7.2.3_20260528` 9.19 GiB.
Сборка 32–48 мин на 4 vCPU, лимит job'а 355 мин.

## Ограничения

- **CI без GPU** — только сборка; приёмка на железе (замеры выше), каждая новая сборка сама не тестируется.
- **Чужие сборки не используются** (`kingj/vllm-openai-rocm-gfx1103`, `krapotkin/vllm-inference-docker-rocm`) —
  артефакт собираем сами из тега vLLM: воспроизводимость и ИБ.
- `gfx1103`-only wheel нерабочий и удалён из релизов — только dual-arch.
- Сборка **не бит-воспроизводима** (ccache/таймстемпы) — сверять с манифестом **своего** прогона.
