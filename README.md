# MLOps Project

## Описание задачи

Проект выполнен в рамках курса **MLOps**.

Цель проекта: автоматизация процесса загрузки данных, предобработки, обучения модели и получения предсказаний через API с использованием Docker и Docker Compose.

ML-задача: предсказание стоимости аренды жилья. Задача регрессии, целевая переменная `price`. Для обучения используется **RandomForestRegressor**.

## Данные

Источник данных: датасет `listings.csv` хранится в **S3** (Yandex Object Storage).

## Структура проекта

- `01_training.ipynb` предобработка данных, обучение и оценка модели;
- `02_pipeline.ipynb` объединение предобработки и модели в sklearn Pipeline;
- `src/train.py` обучение и сохранение Pipeline;
- `src/app.py` FastAPI-приложение для получения предсказаний;
- `Dockerfile` сборка образа для обучения;
- `Dockerfile.api` сборка образа для API;
- `compose.yaml` совместный запуск обучения и API;
- `requirements.txt` зависимости проекта;
- `.gitignore` исключение секретов и артефактов из Git;
- `.dockerignore` исключение секретов и лишних файлов из контекста сборки Docker.

## Настройка S3

Для доступа к данным необходимы действующие ключи S3.

Создать файл `.env` в корне проекта:

```dotenv
KEY_ID=your_key_id
ACCESS_KEY=your_access_key
S3_BUCKET=mlops26-course-bucket
S3_KEY=prepared/listings.csv
S3_ENDPOINT_URL=https://storage.yandexcloud.net
S3_REGION=ru-central1
```

Вместо `your_key_id` и `your_access_key` указываются ключи (файл `.env` не добавляется в Git и Docker-образы).

## Запуск ноутбуков

Ноутбуки подготовлены в Google Colab. Ключи `KEY_ID` и `ACCESS_KEY` добавлены в Colab Secrets.

## Запуск проекта

Необходимо установить и запустить Docker Desktop.

Репозиторий:

```bash
git clone https://github.com/VladaSimon/mlops_project.git
cd mlops_project
```

## Docker Batch

```bash
docker compose up --build train
```

Контейнер автоматически загружает данные из S3, выполняет предобработку, обучение и оценку модели.

Результаты обучения:

- MAE: 66.64
- RMSE: 246.73
- R2: 0.494

Обученный Pipeline сохраняется в `/artifacts/model.pkl` в Docker volume `artifacts`. После обучения контейнер завершается с кодом 0.

## Docker API и Docker Compose

```bash
docker compose up --build
```

Запускается обучение, затем API загружает готовый Pipeline из общего Docker volume.

Документация API: http://localhost:8000/docs

## Пример запроса

POST http://localhost:8000/predict

Тело запроса (JSON):

```json
{
  "latitude": 51.5074,
  "longitude": -0.1278,
  "accommodates": 2,
  "bathrooms": 1,
  "bedrooms": 1,
  "beds": 1,
  "minimum_nights": 2,
  "neighbourhood_cleansed": "Westminster",
  "property_type": "Entire rental unit",
  "room_type": "Entire home/apt"
}
```

Пример ответа (HTTP 200):

```json
{
  "prediction": 331.25313333333327
}
```

Запрос можно отправить через Swagger UI (http://localhost:8000/docs), используя POST /predict.

## Остановка контейнеров

Ctrl+C, затем:

```bash
docker compose down
```
Файл model.pkl создаётся при обучении и не хранится в Git.
