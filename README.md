## Прогнозирование намерения покупки посетителей интернет-магазина

Этот проект направлен на прогнозирование того, совершит ли посетитель интернет-магазина покупку (целевая переменная Revenue). В проекте используются различные методы машинного обучения для классификации, а также проводится анализ данных и сравнение моделей.

## Структура проекта
Проект реализован в Jupyter Notebook `HW4_improved.ipynb` и состоит из нескольких этапов, каждый из которых представлен отдельной ячейкой кода. Ниже описаны этапы, их назначение и результаты.

## 1. Загрузка данных

**Файл:** `data_loader.py`

**Описание:**
Загружает данные с сайта [UCI Machine Learning Repository] и сохраняет их в файл `online_shoppers.csv`. Если файл уже существует, загрузка пропускается.

**Запуск:**
```bash
python data_loader.py
```
**Результаты:**

**Файл** `online_shoppers.csv` с исходными данными.
Лог в файле `data_loader.log`.
Пример вывода:

Файл `online_shoppers.csv` уже существует.

## 2. Очистка данных

**Файл:** `data_cleaning.py` 

**Описание:**
Удаляет дубликаты из данных и сохраняет очищенные данные в файл `cleaned_data.csv`.

**Результаты:**

Файл `cleaned_data.csv` с очищенными данными.
Лог в файле `data_cleaning.log`.
Удалено 125 дубликатов (на основе анализа из следующего этапа).

**Пример вывода**:

Данные очищены и сохранены в `cleaned_data.csv`.

## 3. Исследовательский анализ данных (EDA)

**Файл:** `online_shoppers_eda.py`

**Описание:**
Проводит исследовательский анализ данных (EDA), включая:

Проверку структуры данных и типов столбцов.
Анализ распределения целевой переменной Revenue.
Проверку на пропущенные значения и дубликаты.
Визуализацию распределения числовых признаков.
Построение корреляционной матрицы.
Анализ категориальных признаков (графики для Month, VisitorType, Weekend).

**Результаты:**

Графики распределения числовых признаков в numerical_distributions.png.
Корреляционная матрица в correlation_matrix.png.
Графики распределения категориальных признаков в `Month_distribution.png`, `VisitorType_distribution.png`, `Weekend_distribution.png`.
Лог в файле `online_shoppers_eda.log`.

**Пример вывода**:

Столбцы и типы данных:
Administrative               int64
Administrative_Duration    float64
Informational                int64
Informational_Duration     float64
ProductRelated               int64
ProductRelated_Duration    float64
BounceRates                float64
ExitRates                  float64
PageValues                 float64
SpecialDay                 float64
Month                       object
OperatingSystems             int64
Browser                      int64
Region                       int64
TrafficType                  int64
VisitorType                 object
Weekend                       bool
Revenue                       bool
dtype: object

Распределение Revenue:
Revenue
False    0.845255
True     0.154745
Name: proportion, dtype: float64

Пропущенные значения:
Administrative             0
Administrative_Duration    0
...
Revenue                    0
dtype: int64

Количество дубликатов: 125

## 4. Предобработка данных

**Файл:** `data_preprocessing.py`

**Описание:**
Выполняет предобработку данных:

Кодирование категориальных признаков (Month, VisitorType, Weekend) методом One-Hot Encoding с удалением первого уровня (drop_first=True).
Масштабирование числовых признаков (Administrative_Duration, Informational_Duration, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay) с помощью StandardScaler.
Разделение данных на обучающую (70%) и тестовую (30%) выборки с учетом стратификации по Revenue.
Результаты:

Сохранение объекта scaler в файл `scaler.pkl`.
Лог в файле `data_preprocessing.log`.
Размеры выборок:
Train: (8543, 26)
Test: (3662, 26).

**Пример вывода:**

Размеры выборок:
Train: (8543, 26)
Test: (3662, 26)

## 5. Сравнение моделей

**Файл:** `model_comparison_viz.py`

**Описание:**
Сравнивает производительность 11 моделей машинного обучения:

Gradient Boosting
AdaBoost
Extra Trees
Decision Tree
K Neighbors
SVM Linear
Dummy Classifier
Quadratic Discriminant Analysis (QDA)
LightGBM
XGBoost
CatBoost
Применяется балансировка классов с помощью SMOTE. Оцениваются метрики: Accuracy, Precision, Recall, F1-score, ROC-AUC, а также кросс-валидация (3 фолда).

**Результаты:**

Таблица с результатами всех моделей.
Графики сравнения метрик в файлах `model_comparison_<metric>.png` (например, model_comparison_f1-score.png).
Лог в файле `model_comparison_viz.log`.


## 6. Финальная модель

**Файл:** `final_model.py`

**Описание:**
Использует LightGBM как лучшую модель по итогам сравнения (F1-score = 0.6730, ROC-AUC = 0.9456). Выполняется обучение, оценка метрик и кросс-валидация (5 фолдов). Включает визуализацию сравнения топ-5 моделей по F1-score.

**Результаты:**

Метрики модели LightGBM (конкретные значения зависят от выполнения кода, в предоставленном файле не выведены).
График сравнения топ-5 моделей по F1-score в `top_models_f1_comparison.png`.
Сохранение модели в файл `lightgbm_model.pkl`.
Лог в файле `final_model.log`.

**Пример вывода:**

Итоговая модель: LightGBM
Accuracy: <значение>
Precision: <значение>
Recall: <значение>
F1-score: <значение>
ROC-AUC: <значение>

Кросс-валидация (5 фолдов):
Средний F1-score: <значение>
Стандартное отклонение: <значение>
Все значения F1: [<значения>]
Пример данных для визуализации (из кода):

Модели: Gradient Boosting, LightGBM, CatBoost, XGBoost, AdaBoost
F1-scores: [0.6730, 0.6730, 0.6667, 0.6705, 0.6512]

## 7. Выводы
**Описание:**

Суммирует результаты анализа и моделирования:

Целевая переменная Revenue имеет дисбаланс (85% — False, 15% — True).
Ключевые признаки: PageValues, ExitRates, ProductRelated_Duration.
Лучшая модель: LightGBM с F1-score = 0.6730 и ROC-AUC = 0.9456.
LightGBM демонстрирует высокую точность и сбалансированность между Precision и Recall.

**Как запустить проект**
Установите зависимости:
Убедитесь, что у вас установлен Python 3.10 или выше. Установите необходимые библиотеки:
```bash
pip install -r requirements.txt
```
Создайте файл requirements.txt:
Добавьте следующие строки в файл requirements.txt:

pandas
numpy
scikit-learn
matplotlib
seaborn
lightgbm
xgboost
catboost
imbalanced-learn
joblib

Запустите Jupyter Notebook:
Откройте файл `HW4_improved.ipynb` в Jupyter Notebook или JupyterLab:

```bash
jupyter notebook HW4_improved.ipynb
```
Выполните ячейки последовательно:

Запускайте ячейки по порядку от загрузки данных до выводов. 

**Зависимости**
Для работы проекта необходимы следующие библиотеки:

pandas — обработка данных.
numpy — числовые вычисления.
scikit-learn — предобработка и модели машинного обучения.
matplotlib — визуализация.
seaborn — улучшенные графики.
lightgbm — финальная модель.
xgboost — сравнение моделей.
catboost — сравнение моделей.
imbalanced-learn — балансировка классов (SMOTE).
joblib — сохранение моделей.

## Результаты и файлы
**После выполнения всех этапов вы получите**:

**Данные**: `online_shoppers.csv`, `cleaned_data.csv`.
**Графики**: `numerical_distributions.png`, `correlation_matrix.png`, `Month_distribution.png`, `VisitorType_distribution.png`, `Weekend_distribution.png`, `model_comparison_<metric>.png` (после выполнения), `top_models_f1_comparison.png`.
**Модели**: `scaler.pkl`, `lightgbm_model.pkl`.
**Логи**: `data_loader.log`, `data_cleaning.log`, `online_shoppers_eda.log`, `data_preprocessing.log`, `model_comparison_viz.log`, `final_model.log`.