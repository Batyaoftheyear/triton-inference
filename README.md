# Triton Inference Server: MobileNetV2

Учебный проект по запуску `MobileNetV2` через `Triton Inference Server`.

## Что сделано

- модель: `torchvision MobileNetV2`
- backend: `Python backend`
- протоколы: `HTTP` и `gRPC`
- базовый образ: `nvcr.io/nvidia/tritonserver:23.12-py3`
- кастомные метрики:
  - `custom_processing_time_seconds_total`
  - `custom_requests_in_progress`

## Структура проекта

```text
triton-inference/
  Dockerfile
  requirements.txt
  README.md
  models/
    mobilenet/
      config.pbtxt
      1/
        model.py
  scripts/
    client.py
    run_perf.ps1
    run_model_analyzer.ps1
  analyzer/
    config.yaml
  results/
```

## Вход и выход модели

- вход: `IMAGE`, `FP32`, `[batch, 3, 224, 224]`
- выход: `OUTPUT`, `FP32`, `[batch, 1000]`

## Сборка

```powershell
docker build -t triton-mobilenet .
```

## Запуск

```powershell
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 triton-mobilenet
```

## Проверка готовности

```powershell
curl http://localhost:8000/v2/health/ready
curl http://localhost:8000/v2/models/mobilenet/ready
```

Оба запроса должны вернуть `200 OK`.

## Тестовый запрос

```powershell
python scripts/client.py
```

Клиент отправляет случайный тензор `[1, 3, 224, 224]`.  
Можно передать и реальное изображение:

```powershell
python scripts/client.py --image path\to\image.jpg
```

Если клиент запускается локально на Windows:

```powershell
pip install pillow tritonclient[http] numpy
```

Пример ответа:

```text
Output shape: (1, 1000)
Top-5 class indexes: [21, 92, 127, 22, 128]
Top-5 logits: [3.9488024711608887, 3.788015365600586, 3.340643882751465, 3.1100521087646484, 3.066403388977051]
```

## Метрики

```powershell
curl http://localhost:8002/metrics
```

Нужно проверить наличие строк:

```text
custom_processing_time_seconds_total
custom_requests_in_progress
```

Пример:

```text
custom_processing_time_seconds_total{model="mobilenet"} 32.21110050000607
custom_requests_in_progress{model="mobilenet"} 0
```

## Performance Analyzer

Запуск:

```powershell
.\scripts\run_perf.ps1
```

Результат сохраняется в `results/perf_analyzer.txt`.

Полученные значения:

- throughput: `16.8575 infer/sec`
- average latency: `59281 usec`
- p50 latency: `53572 usec`
- p90 latency: `78256 usec`
- p95 latency: `89463 usec`
- p99 latency: `122275 usec`
- server avg request latency: `36886 usec`
- server compute infer: `35488 usec`

Короткий вывод: на CPU модель работает стабильно, но основное время уходит именно на инференс, поэтому throughput для учебного проекта нормальный, но не высокий.

## Model Analyzer

Запуск:

```powershell
.\scripts\run_model_analyzer.ps1
```

`Model Analyzer` успешно отработал и профилировал `19` конфигураций.

Итоговые файлы:

- `results/model_analyzer/results/metrics-model-inference.csv`
- `results/model_analyzer/reports/summaries/mobilenet/result_summary.html`
- `results/model_analyzer/reports/detailed/mobilenet_config_13/detailed_report.html`
- `results/model_analyzer/reports/detailed/mobilenet_config_14/detailed_report.html`
- `results/model_analyzer/reports/detailed/mobilenet_config_8/detailed_report.html`

Лучшие конфигурации:

- `mobilenet_config_13`: `max_batch_size = 4`, `dynamic_batching = enabled`, `1 CPU instance`, `p99 latency = 104.64 ms`, `throughput = 47.1883 infer/sec`
- `mobilenet_config_14`: `max_batch_size = 8`, `dynamic_batching = enabled`, `1 CPU instance`, `p99 latency = 271.426 ms`, `throughput = 46.0866 infer/sec`
- `mobilenet_config_8`: `max_batch_size = 8`, `dynamic_batching = enabled`, `1 CPU instance`, `p99 latency = 303.453 ms`, `throughput = 45.9609 infer/sec`

Default-конфигурация:

- `p99 latency = 176.84 ms`
- `throughput = 39.9566 infer/sec`

Вывод: лучшей оказалась `mobilenet_config_13`. Она примерно на `18%` лучше default по throughput и при этом даёт более низкую `p99 latency`, чем `mobilenet_config_14` и `mobilenet_config_8`. Для этого проекта самым удачным вариантом оказался режим `max_batch_size = 4`, `dynamic_batching enabled`, `1 CPU instance`.

## Итог

Получился рабочий проект на Triton:

- сервер собирается и запускается в Docker
- модель отвечает на HTTP/gRPC запросы
- клиентский скрипт работает
- кастомные метрики публикуются
- `Performance Analyzer` и `Model Analyzer` отработали, результаты сохранены в `results`
