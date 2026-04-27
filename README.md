# Итоговый проект «Внедрение моделей машинного обучения»

Итоговый проект по дисциплине «Внедрение моделей машинного обучения» для прогнозирования по кредитным картам. Сервис обучает две версии модели на датасете `UCI_Credit_Card.csv`, сохраняет их в `joblib`, поднимает Flask API и демонстрирует основу A/B-тестирования моделей.

## Цель проекта

Разработать и внедрить ML-сервис для бинарной классификации клиентов кредитных карт:
`0` - дефолт в следующем месяце не ожидается;
`1` - дефолт в следующем месяце ожидается.

## Структура репозитория

```
credit-card-default-service/
├── app/                  # Flask API и загрузка моделей
├── data/raw/             # Исходный CSV-датасет
├── models/               # Обученные модели и метрики
├── notebooks/            # Место для исследовательских ноутбуков
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

```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Обучите модели:

```
python -m src.train_model
```

Запустите API:

```
python -m app.api
```

Сервис будет доступен на `http://localhost:5000`.

## Проверка API

Health-check:

```
curl http://localhost:5000/health
```

Пример ответа:

```
{
  "loaded_models": ["v1", "v2"],
  "service": "credit-card-default-service",
  "status": "healthy"
}
```

Предсказание с явным выбором модели:

```
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

Пример ответа:

```
{
  "request_id": "demo-001",
  "prediction": 1,
  "probability": 0.72,
  "model_version": "v1",
  "ab_group": "control",
  "selection_method": "explicit"
}
```

Для A/B-распределения не передавайте `model_version`. Если передать `ab_key`, сервис стабильно назначит клиента в одну из групп:

```
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

## Формат запроса `/predict`

Обязательное поле:

- `features` - объект с 23 числовыми признаками клиента.

Опциональные поля:

- `model_version` - `"v1"` или `"v2"` для ручного выбора модели;
- `request_id` - идентификатор запроса;
- `ab_key` или `customer_id` - ключ для стабильного A/B-распределения.

## Docker

Сборка образа:

```bash
docker build -t <dockerhub-username>/credit-card-default-service:latest .
```

Запуск контейнера:

```bash
docker run --rm -p 5000:5000 <dockerhub-username>/credit-card-default-service:latest
```

Публикация в Docker Hub:

```bash
docker login
docker push <dockerhub-username>/credit-card-default-service:latest
```

Ссылка на Docker Hub после публикации:

```text
https://hub.docker.com/r/<dockerhub-username>/credit-card-default-service
```

## Docker Compose

```bash
docker compose up --build
```

Бонусный профиль с демонстрационным nginx-сервисом:

```bash
docker compose --profile monitoring-demo up --build
```

## Тесты

```bash
pytest
```

## Публикация в GitHub

```bash
git init
git add .
git commit -m "Initial credit card default prediction service"
git branch -M main
git remote add origin <github-repository-url>
git push -u origin main
```

Локальная папка `.venv/` добавлена в `.gitignore` и не должна попадать в репозиторий.

## Метрики модели

После обучения файл `models/metrics.json` содержит:

- Accuracy;
- F1-score для класса дефолта;
- Precision для класса дефолта;
- Recall для класса дефолта;
- ROC-AUC.

Техническая целевая метрика для A/B-теста: `F1-score` по классу дефолта.

## Документация

- [ARCHITECTURE.md](ARCHITECTURE.md) - архитектура, MLOps-концепты, ONNX, uWSGI + NGINX.
- [AB_TEST_PLAN.md](AB_TEST_PLAN.md) - план A/B-тестирования моделей `v1` и `v2`.
