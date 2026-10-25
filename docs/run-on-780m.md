# Запуск собранного wheel на 780M (gfx1103)

Проверять только на железе: CI собирает, но запустить без iGPU не может.

## Переменные окружения

```bash
# gfx1103 нет в fatbin'ах официального torch — притворяемся gfx1100 на уровне HSA.
# На ROCm >= 7.2 этот обход у части пользователей падает SIGSEGV; если так —
# пробовать без него (rocBLAS с 7.1 умеет gfx1103 сам).
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1103
export VLLM_USE_TRITON_FLASH_ATTN=0      # ROCm-FA под RDNA3 нет, идём Triton
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export SDL_VIDEODRIVER=dummy             # APU без дисплея
```

## Старт сервера

```bash
vllm serve <model> \
  --host 0.0.0.0 --port 8000 \
  --enforce-eager \
  --gpu-memory-utilization 0.5 \
  --max-model-len 2048
```

## Почему такие флаги

- `--enforce-eager` — CUDA graph/`torch.compile` на RDNA3-APU нестабильны, а профилировка памяти
  (UMA/GTT вместо VRAM) врёт; eager предсказуемее.
- `--gpu-memory-utilization 0.5` — доля считается от видимой памяти, на APU это GTT из DDR5;
  выше 0.5 начинаются вытеснения и OOM хоста. Для 780M с 8 GiB GTT держать ≤0.4–0.5.
- `--max-model-len 2048` — KV в GTT, пропускная способность DDR5 60–70 ГБ/с; длинный контекст
  не даёт скорости.
- `VLLM_USE_TRITON_FLASH_ATTN=0` — зафиксировано по рабочему рецепту для gfx1103 (см. ниже).

## Проброс устройства (контейнер)

```
--device /dev/kfd --device /dev/dri --group-add video --group-add render
--ipc=host --shm-size 8gb
```

## Источники рецепта

- `krapotkin/vllm-inference-docker-rocm` — рабочий прогон vLLM 0.22.0 на 780M (Ryzen 7 8700G),
  оттуда набор env/флагов.
- `kingj/vllm-openai-rocm-gfx1103` — собранные образы vLLM 0.16/0.17 под gfx1103 (устарели,
  до появления официальных gfx1103-кернелов в v0.20).
- ROCm compatibility matrix — `gfx1103` (Radeon 780M/760M, Phoenix) в поддерживаемых устройствах.

## Ожидания по производительности

APU: 60–70 ГБ/с против 448 ГБ/с у дискретной RTX 5060 Ti. Для 9B в 4-бит expect ~6–10 tok/s —
годится для проверки стека, не для показа.
