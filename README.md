# Favorita Grocery Sales Forecasting

Проект для прогнозирования продаж продуктов с использованием различных моделей временных рядов.

## Структура проекта

```
├── config.py              # Конфигурация проекта
├── run_experiment.py      # Основной скрипт для запуска экспериментов
├── setup_data.py          # Скачивание данных с Google Drive
├── requirements.txt       # Зависимости
├── data/                  # Данные (CSV файлы)
├── src/                   # Исходный код
│   ├── data_loader.py     # Загрузка и предобработка данных
│   ├── features.py        # Инженерия признаков
│   ├── evaluation.py      # Метрики оценки
│   ├── visualization.py   # Визуализация результатов
│   └── models/            # Модели
│       ├── baselines.py   # Базовые модели (Naive, SeasonalNaive, AutoTheta, AutoETS)
│       ├── classical.py   # LightGBM
│       └── neural.py      # Нейронные модели (NHITS, TFT)
└── results/               # Результаты экспериментов
```

## Установка

### 1. Клонирование репозитория

```bash
git clone <repo-url>
cd favorita-grocery-sales-forecasting
```

### 2. Создание виртуального окружения

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

## Подготовка данных

Из-за размера CSV файлы не включены в репозиторий. Для их получения:

### Вариант 1: Автоматическое скачивание (рекомендуется)

```bash
python setup_data.py
```

Этот скрипт скачает архив с Google Drive и распакует его в папку `data/`.

### Вариант 2: Ручная загрузка

Если автоматическое скачивание не работает:
1. Скачайте архив: https://drive.google.com/file/d/1m6OXyjWhBm8Z8RORSjCHNHUpgq_ANXVb/view?usp=sharing
2. Распакуйте в папку `data/`

После распаковки в папке `data/` должны быть файлы:
- `train.csv` — обучающие данные
- `test.csv` — тестовые данные
- `stores.csv` — информация о магазинах
- `items.csv` — информация о товарах
- `oil.csv` — данные о ценах на нефть
- `holidays_events.csv` — праздники и события
- `transactions.csv` — транзакции

## Запуск

Запуск всех моделей (базовые, LightGBM, нейронные):

```bash
python run_experiment.py
```

Результаты сохраняются в директорию `results/`:
- `metrics.csv` — таблица метрик по всем моделям
- `metrics.json` — метрики в JSON формате
- `metrics_comparison.png` — график сравнения метрик
- `forecasts/` — примеры прогнозов

## Конфигурация

Основные параметры в `config.py`:

| Параметр | Описание | Значение по умолчанию |
|----------|----------|----------------------|
| `N_SERIES` | Количество временных рядов | 150 |
| `TRAIN_END` | Конец обучающего периода | "2017-07-15" |
| `VAL_END` | Конец валидационного периода | "2017-07-31" |
| `HORIZON` | Горизонт прогнозирования | 16 |
| `MAX_EPOCHS` | Максимальное число эпох | 50 |
| `BATCH_SIZE` | Размер батча | 32 |

## Модели

Проект включает следующие модели:

1. **Базовые модели**:
   - Naive
   - SeasonalNaive
   - AutoTheta
   - AutoETS

2. **Классическая ML**:
   - LightGBM

3. **Нейронные модели**:
   - NHITS
   - TFT (Temporal Fusion Transformer)

## Метрики

Используемые метрики оценки:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- RMSLE (Root Mean Squared Logarithmic Error)
- Weighted RMSE (с учётом perishable товаров)
