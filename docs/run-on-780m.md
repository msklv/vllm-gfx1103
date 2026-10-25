# Запуск собранного wheel на 780M (gfx1103)

Всё ниже — **проверено на железе** 2026-10-25 (Radeon 780M, узел k8s-04), прогон в поде на
`rocm/vllm-dev:base` с проброшенными `/dev/kfd` + `/dev/dri`.

## Обязательные условия

1. **Колесо dual-arch** (`PYTORCH_ROCM_ARCH="gfx1103;gfx1100"`). Колесо только под `gfx1103` даёт
   `hipErrorInvalidImage` в режиме override, а без override падает сам torch — см. README.
2. **`HSA_OVERRIDE_GFX_VERSION=11.0.0`** — у официального torch нет кодобъектов gfx1103, без override
   `torch.randn` даёт `hipErrorInvalidKernelFile`.
3. **`privileged: true`** в поде (одного `runAsUser: 0` + hostPath недостаточно: device-cgroup не пустит,
   и `torch.cuda.is_available()` вернёт `False` без ошибки).

## Старт сервера (рабочая комбинация)

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
python3 -m vllm.entrypoints.openai.api_server \
  --model /work/models/qwen3-0.6b --served-model-name qwen3-0.6b --port 8000 \
  --enforce-eager --max-model-len 1024 --gpu-memory-utilization 0.25 \
  --no-enable-prefix-caching
```

Замер на `Qwen/Qwen3-0.6B`: веса 1.12 GiB / 1.96 s, KV 1.89 GiB (17 680 токенов), GTT 4.13 GB
из 15.13, генерация **59.9–60.9 tok/s** (русский), 4 запроса подряд без ошибок.

## Чего не делать (проверено, ломается)

| Конфиг | Что происходит |
|---|---|
| `--max-model-len 2048 --gpu-memory-utilization 0.35` + prefix-caching по умолчанию | после загрузки весов `HW Exception … GPU Hang`, ядро делает `amdgpu: GPU reset` |
| колесо только с `gfx1103` + `HSA_OVERRIDE_GFX_VERSION=11.0.0` | `hipErrorInvalidImage` на первом vLLM-ядре |
| любой запуск без `HSA_OVERRIDE_GFX_VERSION` | `hipErrorInvalidKernelFile` на `torch.randn` |
| `VLLM_ATTENTION_BACKEND=TRITON_ATTN` | в 0.25.1 игнорируется: в логе `Overriding with ROCM_ATTN out of potential backends` |

## Проброс устройства

```yaml
securityContext: {privileged: true, runAsUser: 0}
volumeMounts:
  - {name: dri, mountPath: /dev/dri}
  - {name: kfd, mountPath: /dev/kfd}
volumes:
  - {name: dri, hostPath: {path: /dev/dri}}
  - {name: kfd, hostPath: {path: /dev/kfd, type: CharDevice}}
```

ROCm-утилиты в образе не в `PATH`: `/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel/bin/{rocminfo,rocm-smi,amd-smi}`.

## Ожидания по производительности

APU: ~60–70 ГБ/с против 448 ГБ/с у дискретной RTX 5060 Ti. Замерено: 0.6B — **60 tok/s**;
9B в 4-бит — оценочно 6–10 tok/s (для проверки стека, не для показа).

## Источники рецепта

- `krapotkin/vllm-inference-docker-rocm` — прогон vLLM 0.22.0 на 780M (Ryzen 7 8700G), оттуда набор env/флагов.
- `kingj/vllm-openai-rocm-gfx1103` — образы vLLM 0.16/0.17 под gfx1103 (устарели, до появления gfx1103-ядер в v0.20).
- ROCm compatibility matrix — `gfx1103` (Radeon 780M/760M, Phoenix) в поддерживаемых устройствах.
