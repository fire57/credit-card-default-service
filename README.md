# Итоговый проект «Внедрение моделей машинного обучения»

## Описание проекта

Проект посвящен разработке и внедрению production-like ML-сервиса для прогнозирования дефолта по кредитным картам.

Сервис обучает две версии модели бинарной классификации на датасете `UCI_Credit_Card.csv`, сохраняет модели в формате `joblib`, поднимает Flask API для инференса и демонстрирует возможность A/B-тестирования моделей `v1` и `v2`.

Домен проекта: финансы / кредитный скоринг.

Целевая переменная:

- `0` - дефолт в следующем месяце не ожидается;
- `1` - дефолт в следующем месяце ожидается.

## Цели проекта

- Подготовить ML-модель к использованию в production-like среде.
- Реализовать API-сервис с эндпоинтами `/health` и `/predict`.
- Сохранить и загрузить модели через `joblib`.
- Обеспечить воспроизводимый запуск через `requirements.txt`.
- Упаковать сервис в Docker-контейнер.
- Описать архитектурные решения и MLOps-концепты.
- Реализовать практическую демонстрацию A/B-тестирования двух версий модели.

## Структура проекта

```text
credit-card-default-service/
├── app/                  # Flask API и загрузка моделей
├── data/raw/             # Исходный датасет
├── demo                  # Скриншоты из Postman для демонстрации             
├── models/               # Обученные модели и метрики
├── src/                  # Код обучения и подготовки данных
├── tests/                # Тесты API
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── ARCHITECTURE.md
├── AB_TEST_PLAN.md
└── README.md
```

## Локальный запуск

Создайте и активируйте виртуальное окружение:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Установите зависимости:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Обучите модели:

```bash
python -m src.train_model
```

После обучения в папке `models/` будут созданы:

```text
model_v1.joblib
model_v2.joblib
metrics.json
```

Запустите Flask API:

```bash
python -m app.api
```

По умолчанию сервис доступен по адресу:

```text
http://localhost:5000
```

Если порт `5000` занят, запустите сервис на другом порту:

```bash
PORT=5051 python -m app.api
```

Тогда запросы нужно отправлять на:

```text
http://localhost:5051
```

## Запуск в Docker

Сборка Docker-образа:

```bash
docker build -t fire57/credit-card-default-service:latest .
```

Запуск контейнера:

```bash
docker run --rm -p 5000:5000 fire57/credit-card-default-service:latest
```

Проверка контейнера:

```bash
curl http://localhost:5000/health
```

Запуск через Docker Compose:

```bash
docker compose up --build
```

Бонусный профиль с демонстрационным nginx-сервисом:

```bash
docker compose --profile monitoring-demo up --build
```

## Docker Hub

Docker-образ после публикации доступен по ссылке:

```text
https://hub.docker.com/r/fire57/credit-card-default-service
```

Команды для публикации образа:

```bash
docker login
docker push fire57/credit-card-default-service:latest
```

## API

Сервис предоставляет два эндпоинта:

```text
GET /health
POST /predict
```

## `GET /health`

Эндпоинт проверяет работоспособность сервиса и наличие загруженных моделей.

Пример запроса:

```bash
curl http://localhost:5000/health
```

Пример ответа:

```json
{
  "loaded_models": ["v1", "v2"],
  "service": "credit-card-default-service",
  "status": "healthy"
}
```

Поля ответа:

- `status` - состояние сервиса;
- `service` - название сервиса;
- `loaded_models` - список загруженных версий моделей.

## `POST /predict`

Эндпоинт принимает JSON с признаками клиента и возвращает прогноз дефолта.

### Формат запроса

```json
{
  "request_id": "demo-001",
  "model_version": "v1",
  "features": {
    "LIMIT_BAL": 20000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 24,
    "PAY_0": 2,
    "PAY_2": 2,
    "PAY_3": -1,
    "PAY_4": -1,
    "PAY_5": -2,
    "PAY_6": -2,
    "BILL_AMT1": 3913,
    "BILL_AMT2": 3102,
    "BILL_AMT3": 689,
    "BILL_AMT4": 0,
    "BILL_AMT5": 0,
    "BILL_AMT6": 0,
    "PAY_AMT1": 0,
    "PAY_AMT2": 689,
    "PAY_AMT3": 0,
    "PAY_AMT4": 0,
    "PAY_AMT5": 0,
    "PAY_AMT6": 0
  }
}
```

Обязательное поле:

- `features` - объект с 23 числовыми признаками клиента.

Опциональные поля:

- `request_id` - идентификатор запроса;
- `model_version` - версия модели: `"v1"` или `"v2"`;
- `ab_key` или `customer_id` - ключ для стабильного A/B-распределения.

Если `model_version` не передан, сервис автоматически распределяет запрос между моделями `v1` и `v2`.

Если передан `ab_key`, один и тот же клиент будет стабильно попадать в одну и ту же A/B-группу.

### Формат ответа

```json
{
  "request_id": "demo-001",
  "prediction": 1,
  "probability": 0.72,
  "model_version": "v1",
  "ab_group": "control",
  "selection_method": "explicit"
}
```

Поля ответа:

- `request_id` - идентификатор запроса;
- `prediction` - прогноз модели: `0` или `1`;
- `probability` - вероятность дефолта;
- `model_version` - версия использованной модели;
- `ab_group` - A/B-группа: `control` или `treatment`;
- `selection_method` - способ выбора модели.

## Примеры запросов к API

### Предсказание через модель `v1`

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "demo-v1",
    "model_version": "v1",
    "features": {
      "LIMIT_BAL": 20000,
      "SEX": 2,
      "EDUCATION": 2,
      "MARRIAGE": 1,
      "AGE": 24,
      "PAY_0": 2,
      "PAY_2": 2,
      "PAY_3": -1,
      "PAY_4": -1,
      "PAY_5": -2,
      "PAY_6": -2,
      "BILL_AMT1": 3913,
      "BILL_AMT2": 3102,
      "BILL_AMT3": 689,
      "BILL_AMT4": 0,
      "BILL_AMT5": 0,
      "BILL_AMT6": 0,
      "PAY_AMT1": 0,
      "PAY_AMT2": 689,
      "PAY_AMT3": 0,
      "PAY_AMT4": 0,
      "PAY_AMT5": 0,
      "PAY_AMT6": 0
    }
  }'
```

### Предсказание через модель `v2`

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "demo-v2",
    "model_version": "v2",
    "features": {
      "LIMIT_BAL": 20000,
      "SEX": 2,
      "EDUCATION": 2,
      "MARRIAGE": 1,
      "AGE": 24,
      "PAY_0": 2,
      "PAY_2": 2,
      "PAY_3": -1,
      "PAY_4": -1,
      "PAY_5": -2,
      "PAY_6": -2,
      "BILL_AMT1": 3913,
      "BILL_AMT2": 3102,
      "BILL_AMT3": 689,
      "BILL_AMT4": 0,
      "BILL_AMT5": 0,
      "BILL_AMT6": 0,
      "PAY_AMT1": 0,
      "PAY_AMT2": 689,
      "PAY_AMT3": 0,
      "PAY_AMT4": 0,
      "PAY_AMT5": 0,
      "PAY_AMT6": 0
    }
  }'
```

### A/B-распределение без явного выбора модели

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ab_key": "customer-123",
    "features": {
      "LIMIT_BAL": 20000,
      "SEX": 2,
      "EDUCATION": 2,
      "MARRIAGE": 1,
      "AGE": 24,
      "PAY_0": 2,
      "PAY_2": 2,
      "PAY_3": -1,
      "PAY_4": -1,
      "PAY_5": -2,
      "PAY_6": -2,
      "BILL_AMT1": 3913,
      "BILL_AMT2": 3102,
      "BILL_AMT3": 689,
      "BILL_AMT4": 0,
      "BILL_AMT5": 0,
      "BILL_AMT6": 0,
      "PAY_AMT1": 0,
      "PAY_AMT2": 689,
      "PAY_AMT3": 0,
      "PAY_AMT4": 0,
      "PAY_AMT5": 0,
      "PAY_AMT6": 0
    }
  }'
```

## Тесты

Запуск тестов:

```bash
pytest
```

## Метрики модели

После обучения файл `models/metrics.json` содержит:

- Accuracy;
- F1-score для класса дефолта;
- Precision для класса дефолта;
- Recall для класса дефолта;
- ROC-AUC.

Техническая целевая метрика для A/B-теста: `F1-score` для класса дефолта.

## Дополнительная документация

- [ARCHITECTURE.md](ARCHITECTURE.md) - архитектура сервиса, MLOps-концепты, ONNX, uWSGI + NGINX.
- [AB_TEST_PLAN.md](AB_TEST_PLAN.md) - план A/B-тестирования моделей `v1` и `v2`.
