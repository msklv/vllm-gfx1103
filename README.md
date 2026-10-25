# vllm-gfx1103

> **Статус: ✅ проверено на реальном железе** — 2026-10-25, Radeon 780M (`gfx1103`), узел k8s-04:
> vLLM 0.25.1 на ROCm 10.0.0, **59.9–60.9 tok/s** на Qwen3-0.6B, 4 запроса подряд без ошибок.
> На том же колесе проверены pooling-задачи: эмбеддинги `bge-m3` — **2841 tok/s**, реранк
> `bge-reranker-v2-m3` — **515 мс на 4 пары** (`/v1/embeddings`, `/rerank`).
> Артефакт: релиз [`build-v0.25.1-rocm10.0.0-gfx1103-gfx1100`](https://github.com/msklv/vllm-gfx1103/releases/tag/build-v0.25.1-rocm10.0.0-gfx1103-gfx1100)
> + git-тег `v0.25.1-rocm10.0.0-gfx1103-gfx1100-verified`.

Сборка vLLM под ROCm для **AMD Radeon 780M / 760M (Phoenix APU, LLVM target `gfx1103`)** в GitHub Actions,
без GPU на раннере (HIP компилируется headless). Апстрим официально не публикует ни образов, ни колёс под
`gfx1103` (в списке поддержки `gfx90a/942/950`, дискретные `gfx1100/1101/1102`, `gfx1200/1201`,
Ryzen AI 300/Max `gfx1150/1151`), хотя код под эту архитектуру в vLLM есть с PR #40037
(влит 2026-04-23, первый релиз — v0.20.0).

## Артефакт

| Что | Значение |
|---|---|
| Wheel | `vllm-0.25.1+rocm100-cp312-cp312-linux_x86_64.whl`, **68.5 MB** |
| sha256 | `cd53886a0540b9432de3e113b241df66191b5afe90fbebdd1541a6c9d018e688` (сверять с `MANIFEST.txt`) |
| Релиз | `build-v0.25.1-rocm10.0.0-gfx1103-gfx1100` (бессрочно, ассеты не истекают) |
| Git-тег | `v0.25.1-rocm10.0.0-gfx1103-gfx1100-verified` (аннотированный, на коммите сборки) |
| Сборка | run [40194481635](https://github.com/msklv/vllm-gfx1103/actions/runs/40194481635), 48 мин на 4 vCPU |
| Источник | vLLM `v0.25.1`, commit `752a3a504485790a2e8491cacbb35c137339ad34` |
| База | `rocm/vllm-dev:base` → torch **2.13.0+rocm10.0.0**, HIP 7.15.26333 (ROCm 10.0.0) |
| Arch | `PYTORCH_ROCM_ARCH="gfx1103;gfx1100"` → в fatbin `10 × amdhsa--gfx1100` + `10 × amdhsa--gfx1103` |
| Требования | Python **3.12** (`cp312`), ROCm **10.0.0** runtime, `torch 2.13.0+rocm10.0.0`, Apache-2.0 |

Скачать и проверить:

```bash
gh release download build-v0.25.1-rocm10.0.0-gfx1103-gfx1100 -R msklv/vllm-gfx1103 -D ./wheel
sha256sum ./wheel/*.whl   # сверить с строкой sha256: из ./wheel/MANIFEST.txt
```

## Замеры на железе (2026-10-25, Radeon 780M, k8s-04)

Модель для приёмочного теста — **`Qwen/Qwen3-0.6B`** (0.6B, 1.5 GB, `Qwen3ForCausalLM`): выбрана как
минимальная проверка стека «вес → KV → генерация», качается за 2.5 мин на канале площадки.

| Параметр | Результат |
|---|---|
| `rocminfo` | `gfx1103`, `amdgcn-amd-amdhsa--gfx1103`, AMD Radeon 780M Graphics |
| torch | `avail True`, `gcnArchName: gfx1103`, `total_memory` = GTT **14.09 GiB** |
| Колесо | скачано подом с GitHub, 6.3 MB/s, sha256 = MANIFEST ✅ |
| vLLM | `0.25.1`, загрузка весов **1.12 GiB / 1.96 s** |
| KV-кэш | 1.89 GiB → **17 680 токенов** (`--max-model-len 1024`) |
| GTT после загрузки | **4.13 GB** из 15.13 (RAM ноды 7 GiB из 28) |
| `torch.ops._C.rms_norm` (ядро из колеса) | ✅ в обоих режимах HSA |
| Генерация (русский) | **59.9–60.9 tok/s**, стабильно на 4 запросах подряд |
| GPU reset | 1 при неудачном конфиге; после подбора флагов — 0 |

## Эмбеддинги и реранк: тоже проверено на 780M

vLLM 0.25.1 в этом колесе умеет pooling-задачи, отдельный llama.cpp не нужен. Задачи задаются
`--runner pooling` + `--convert {embed,classify}` (старый `--task` в 0.25 убран).

| Роль | Модель | Запуск | Эндпоинты | Замер на 780M |
|---|---|---|---|---|
| **Эмбеддинги** | `BAAI/bge-m3` (fp16, 1.14 GiB) | `--runner pooling --convert embed` | `/v1/embeddings` (OpenAI), `/v2/embed` (Cohere), `/pooling` | dim **1024**; 3202 токена за 1.13 с → **2841 tok/s** (3 прогона подряд, прогрев 3.0 с) |
| **Реранк** | `BAAI/bge-reranker-v2-m3` (fp16, 1.14 GiB) | `--runner pooling --convert classify` | `/rerank`, `/v1/rerank`, `/v2/rerank` (Cohere), `/score` | 4 пары за **515 мс** (≈7.8 пар/с); скоры `0.9972` на релевантном документе, `0.0001`/`0.0` на посторонних |

Особенности, которые важны на практике:

- реранк работает как scoring-модель **только при `num_labels == 1`** (у `bge-reranker-v2-m3` так и есть) —
  тогда vLLM включает и `/score`, и Cohere-совместимый `/rerank`, отдельный адаптер для RAGFlow не нужен;
- у pooling-моделей **нет KV-кэша**: `--gpu-memory-utilization 0.20` на каждую и достаточно,
  фактический GTT ≈ **1.55 GB** на модель (два сервера + страж-0.6B укладываются в ~5.2 GB из 15.13);
- веса нужны **в safetensors** (GGUF vLLM не грузит): на HF `bge-m3` лежит в fp32 (2.27 GB), после
  конвертации в fp16 — 1.14 GB;
- требования те же, что для генерации: колесо с двумя архитектурами, `HSA_OVERRIDE_GFX_VERSION=11.0.0`,
  `privileged: true`, `--enforce-eager`.

## Обязательно: сборка под ДВЕ архитектуры

Колесо, собранное только под `gfx1103`, на этом железе **не работает** — проверены оба режима:

| Режим | `torch.randn` (ядра torch) | `torch.ops._C.rms_norm` (ядра vLLM) |
|---|---|---|
| без `HSA_OVERRIDE_GFX_VERSION` | ❌ `hipErrorInvalidKernelFile` | не доходит (torch падает раньше) |
| `HSA_OVERRIDE_GFX_VERSION=11.0.0` | ✅ (устройство рапортует `gfx1100`) | ❌ `hipErrorInvalidImage` — в fatbin только `amdhsa--gfx1103` |

Причина: официальный torch в `rocm/vllm-dev:base` собран без кодобъектов `gfx1103`. Поэтому нужны оба
набора — `gfx1100` для режима override (torch + vLLM) и `gfx1103` на будущее, когда torch начнут
собирать под Phoenix:

```bash
gh workflow run build-wheel.yml -R msklv/vllm-gfx1103 \
  -f vllm_ref=v0.25.1 -f 'arch=gfx1103;gfx1100' -f max_jobs=4 \
  -f release_tag=build-v0.25.1-rocm10.0.0-gfx1103-gfx1100
```

## Как запустить на APU

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0        # обязателен из-за torch
python3 -m vllm.entrypoints.openai.api_server \
  --model <model> --served-model-name <name> --port 8000 \
  --enforce-eager --max-model-len 1024 --gpu-memory-utilization 0.25 \
  --no-enable-prefix-caching
```

⚠️ `--max-model-len 2048 --gpu-memory-utilization 0.35` с prefix-caching по умолчанию → **GPU Hang**
и сброс драйвера (`amdgpu: GPU reset`) на профилировке KV. На APU этот конфиг не использовать.
Подробнее — `docs/run-on-780m.md`.

## Что делает workflow

`.github/workflows/build-wheel.yml` (ручной запуск, `workflow_dispatch`, входы `vllm_ref`, `base_image`,
`arch`, `max_jobs`, `publish_release`, `release_tag`):

1. освобождает диск раннера (~30 GB) — образ базы ROCm 15–25 GB;
2. клонирует `vllm-project/vllm` на заданный тег;
3. печатает версию ROCm (из `torch.__version__`) и torch/hip базового образа, падает если версия не определена;
4. собирает **только wheel** через апстримный `docker/Dockerfile.rocm`, таргет `export_vllm`, с
   `ARG_PYTORCH_ROCM_ARCH=<arch>` — без 30-гигабайтного runtime-образа;
5. формирует `MANIFEST.txt` (sha256, размер, источник, база, ROCm/torch, arch, номер прогона);
6. публикует wheel + манифест в GitHub Release (`--clobber` при повторе) и в артефакт на 30 дней.

Код vLLM не патчится: `export_vllm` — официальный «dist»-таргет Dockerfile.rocm, меняется только архитектура.

Базовые образы (`rocm/vllm-dev`): `base` — 15.63 GiB, дефолт; `base_custom_base_rocm7.2.3_20260528` —
9.19 GiB, если упирается диск. Время сборки: 32–48 мин на 4 vCPU, лимит job'а 355 мин.

## Ограничения

- **Раннер без GPU**: CI только собирает. Приёмочный прогон сделан на железе (см. замеры), но каждая
  новая сборка сама по себе не тестируется.
- **Чужие сборки не используются** (`kingj/vllm-openai-rocm-gfx1103`, рецепт `krapotkin/vllm-inference-docker-rocm`) —
  берём их идеи, но артефакт собираем сами из тега vLLM: воспроизводимость и ИБ.
- **Колесо `gfx1103`-only нерабочее** и удалено из релизов — оставлен только dual-arch артефакт.
- Сборка **не бит-воспроизводима** (ccache/таймстемпы): sha256 у каждого прогона свой, сверять с манифестом
  **того же** прогона.
