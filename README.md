# vllm-gfx1103

Сборка vLLM под ROCm для **AMD Radeon 780M / 760M (Phoenix APU, LLVM target `gfx1103`)** — чисто CI-сборка,
без GPU на раннере. Апстрим официально не публикует ни образов, ни колёс для `gfx1103` (в списке поддержки
только `gfx90a/942/950`, дискретные `gfx1100/1101/1102`, `gfx1200/1201`, Ryzen AI 300/Max `gfx1150/1151`),
хотя код под эту архитектуру в vLLM есть с PR #40037 (влит 2026-04-23, первый релиз — v0.20.0).

## Что делает workflow

`.github/workflows/build-wheel.yml` (ручной запуск, `workflow_dispatch`):

1. освобождает диск раннера (~30 GB) — образ базы ROCm весит 15–25 GB;
2. клонирует `vllm-project/vllm` на заданный тег;
3. печатает версию ROCm и torch/hip базового образа (нужно, чтобы убедиться: ROCm ≥ 7.1 — с неё
   rocBLAS получает Tensile-логику под `gfx1103`);
4. собирает **только wheel** через апстримный `docker/Dockerfile.rocm`, таргет `export_vllm`,
   с `ARG_PYTORCH_ROCM_ARCH=gfx1103` — без сборки 30-гигабайтного runtime-образа;
5. кладёт `*.whl` в артефакт сборки.

Соответствие апстриму: `export_vllm` — это официальный «dist»-таргет Dockerfile.rocm (в нём же лежат
`requirements/`, `tests/`, `examples/`, docker-файлы). Мы не патчим код vLLM — только задаём архитектуру.

## Результат сборки №1 (2026-10-25)

| Параметр | Значение |
|---|---|
| Run | [40150282601](https://github.com/msklv/vllm-gfx1103/actions/runs/40150282601) — `success` |
| Длительность | **32 мин 12 с** (01:21:02 → 01:53:14 UTC), 4 vCPU |
| Источник | vLLM `v0.25.1`, commit `752a3a504485790a2e8491cacbb35c137339ad34` |
| База | `rocm/vllm-dev:base` → **torch 2.13.0+rocm10.0.0**, **HIP 7.15.26333** (ROCm 10.0.0) |
| Артефакт | `vllm-0.25.1+rocm100-cp312-cp312-linux_x86_64.whl`, 48 MiB (205 MiB распакованный), Apache-2.0 |
| Arch | лог сборки: `-- HIP architectures: gfx1103`, `Building PyTorch for GPU arch: gfx1103` |
| Проверка бинаря | `_rocm_C.abi3.so` (65.4 MiB): секция `.hip_fatbin` есть, ISA `amdhsa--gfx1103` × 4, `gfx942/gfx1100/gfx1201` — 0 → код только под нашу архитектуру |

Требования на целевой машине: **Python 3.12** (wheel `cp312`), runtime **ROCm 10.0.0** (под него собран
и линкуется), `torch 2.13.0+rocm10.0.0`.

## Где лежит результат

| Место | Срок жизни |
|---|---|
| **GitHub Release** `build-<vllm_ref>-rocm<ver>-<arch>` — [build-v0.25.1-rocm10.0.0-gfx1103](https://github.com/msklv/vllm-gfx1103/releases/tag/build-v0.25.1-rocm10.0.0-gfx1103) | **бессрочно** (ассеты релиза не истекают) |
| Артефакт прогона (Actions → run → Artifacts) | 30 дней |

В релиз кладутся `*.whl` и `MANIFEST.txt` (sha256, размер, источник + commit, база, ROCm/torch, arch,
номер прогона, требования к рантайму). Повторный запуск с теми же `vllm_ref`/`arch` **перезаписывает
ассеты того же тега** (`--clobber`), поэтому тег всегда указывает на последнюю удачную сборку этой
комбинации; для параллельных вариантов есть вход `release_tag`.

Скачать:

```bash
gh release download build-v0.25.1-rocm10.0.0-gfx1103 -R msklv/vllm-gfx1103 -D ./wheel
sha256sum -c <(grep sha256 ./wheel/MANIFEST.txt | awk '{print $2"  ./wheel/"$1}')
```


## Запуск

```bash
gh workflow run build-wheel.yml -R msklv/vllm-gfx1103 \
  -f vllm_ref=v0.25.1 -f base_image=rocm/vllm-dev:base -f arch=gfx1103 -f max_jobs=4
gh run list -R msklv/vllm-gfx1103 --limit 3
gh run watch <run-id> -R msklv/vllm-gfx1103
```

## Проверено на железе (2026-10-25, Radeon 780M, k8s-04)

| Параметр | Результат |
|---|---|
| `rocminfo` | `gfx1103`, `amdgcn-amd-amdhsa--gfx1103`, AMD Radeon 780M Graphics |
| torch | `avail True`, `gcnArchName: gfx1103` (без override), `total_memory` = GTT 14.09 GiB |
| Колесо из релиза | скачано подом с GitHub, 6.3 MB/s, sha256 = MANIFEST ✅ |
| vLLM | `0.25.1`, `Qwen3ForCausalLM`, загрузка весов **1.12 GiB / 1.96 s** |
| KV-кэш | 1.89 GiB → **17 680 токенов** при `--max-model-len 1024` |
| GTT после загрузки | **4.13 GB** из 15.13 (RAM ноды: 7 GiB used из 28) |
| Генерация (рус.) | **59.9–60.9 tok/s**, стабильно на 4 запросах подряд |
| GPU reset | 1 раз при неудачном конфиге, после подбора флагов — 0 |

### Обязательное: сборка под ДВЕ архитектуры

Чистое `gfx1103` колесо на APU **не работает**, проверено оба режима:

| Режим | `torch.randn` (ядра torch) | `torch.ops._C.rms_norm` (наши ядра) |
|---|---|---|
| без `HSA_OVERRIDE_GFX_VERSION` | ❌ `hipErrorInvalidKernelFile` | не проверяется (torch падает раньше) |
| `HSA_OVERRIDE_GFX_VERSION=11.0.0` | ✅ (устройство рапортует `gfx1100`) | ❌ `hipErrorInvalidImage` — в fatbin только `amdhsa--gfx1103` |

Причина: официальный torch в `rocm/vllm-dev:base` собран без кодобъектов gfx1103. Поэтому:

```bash
gh workflow run build-wheel.yml -R msklv/vllm-gfx1103 \
  -f vllm_ref=v0.25.1 -f 'arch=gfx1103;gfx1100' -f max_jobs=4 \
  -f release_tag=build-v0.25.1-rocm10.0.0-gfx1103-gfx1100
```

Dual-arch колесо (68.5 MB, fatbin `10 × gfx1100` + `10 × gfx1103`) с `HSA_OVERRIDE_GFX_VERSION=11.0.0`
даёт рабочий vLLM: torch и наши ядра оба идут gfx1100-кодом.

### Рабочая команда на APU

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
python3 -m vllm.entrypoints.openai.api_server \
  --model <model> --served-model-name <name> --port 8000 \
  --enforce-eager --max-model-len 1024 --gpu-memory-utilization 0.25 \
  --no-enable-prefix-caching
```

`--max-model-len 2048 --gpu-memory-utilization 0.35` (с prefix-caching по умолчанию) привёл к
**GPU Hang** и сбросу драйвера на профилировке KV — этот конфиг на APU не использовать.


Кандидаты базового образа (Docker Hub `rocm/vllm-dev`):

| Тег | Размер | Заметка |
|---|---|---|
| `base` | 15.63 GiB | дефолт Dockerfile.rocm, обновлён 2026-10-20 |
| `base_custom_base_rocm7.2.3_20260528` | 9.19 GiB | явно ROCm 7.2.3 — меньше по диску, если сборка упирается в место |

Ожидаемое время: 1–3 часа на 4 vCPU (HIP/C++ + Rust-стадия), лимит job'а — 355 минут.

## Ограничения (важно)

- **Раннер GitHub без GPU** — собираем, но не запускаем: компиляция HIP не требует устройства, а
  smoke-тест возможен только на самом 780M.
- **Чужие сборки не используются** (в т.ч. `kingj/vllm-openai-rocm-gfx1103` и рецепт
  `krapotkin/vllm-inference-docker-rocm` из 2026-06) — берём их идеи, но артефакт собираем сами
  из тега vLLM, чтобы он был воспроизводим и проходил ИБ.
- Wheel ещё не проверен на железе. Установка/запуск на 780M — см. `docs/run-on-780m.md`.
