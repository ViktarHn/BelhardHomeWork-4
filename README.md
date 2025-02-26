# BelhardHomeWork-4
Прогнозирование намерения покупки посетителей интернет-магазина

Этот проект направлен на прогнозирование того, совершит ли посетитель интернет-магазина покупку (целевая переменная Revenue). В проекте используются различные методы машинного обучения для классификации, а также проводится анализ данных и сравнение моделей.

Структура проекта

Проект состоит из нескольких этапов, каждый из которых выполняется отдельным скриптом. Ниже описано, что делает каждый файл и в каком порядке их нужно запускать.

1. Загрузка данных

Файл: data_loader.py

Описание: Загружает данные с сайта UCI Machine Learning Repository и сохраняет их в файл online_shoppers.csv.

Запуск:

bash

python data_loader.py

2. Анализ данных (EDA)

Файл: online_shoppers_eda.py

Описание: Проводит исследовательский анализ данных (EDA), включая:

Проверку структуры данных.

Распределение целевой переменной Revenue.

Проверку на пропущенные значения и дубликаты.

Визуализацию распределения числовых признаков.

Построение корреляционной матрицы.

Результаты:

Графики распределения числовых признаков сохраняются в numerical_distributions.png.

Корреляционная матрица сохраняется в correlation_matrix.png.

Запуск:

bash

python online_shoppers_eda.py

3. Очистка данных
   
Файл: data_cleaning.py

Описание: Удаляет дубликаты из данных и сохраняет очищенные данные в файл cleaned_data.csv.

Запуск:

bash

python data_cleaning.py

4. Предобработка данных

Файл: data_preprocessing.py

Описание: Выполняет предобработку данных, включая:
Кодирование категориальных признаков (One-Hot Encoding).
Масштабирование числовых признаков.
Разделение данных на обучающую и тестовую выборки.

Запуск:

bash

python data_preprocessing.py

5. Обучение и оценка моделей

Описание: В этом этапе обучаются различные модели машинного обучения, и для каждой модели вычисляются метрики (Accuracy, Precision, Recall, F1-score). Также выполняется кросс-валидация для оценки устойчивости моделей.
Файлы:

Gradient Boosting: cl_gradientbc.py

CatBoost: cl_catboost.py

AdaBoost: cl_adaboost.py

LightGBM: cl_lightGBM.py

XGBoost: cl_xgboost.py

Запуск (для каждой модели):
bash

python cl_gradientbc.py

python cl_catboost.py

python cl_adaboost.py

python cl_lightGBM.py

python cl_xgboost.py


6. Сравнение моделей

Файл: model_comparison_viz.py
Описание: Сравнивает все обученные модели по метрикам (Accuracy, Precision, Recall, F1-score) и визуализирует результаты.

Результаты:

Графики сравнения моделей по каждой метрике сохраняются в файлы:

model_comparison_accuracy.png

model_comparison_precision.png

model_comparison_recall.png

model_comparison_f1-score.png

Запуск:

bash

python model_comparison_viz.py

7. Выбор и сохранение лучшей модели

Файл: final_model.py

Описание:

Выбирает лучшую модель на основе сравнения метрик (в данном случае LightGBM), сохраняет её в файл lightgbm_model.pkl и визуализирует сравнение топ-5 моделей по F1-score.

Результаты:

Лучшая модель сохраняется в lightgbm_model.pkl.
График сравнения топ-5 моделей по F1-score сохраняется в top_models_f1_comparison.png.

Запуск:

bash

python final_model.py

Результаты

После выполнения всех этапов вы получите:

Очищенные и предобработанные данные в файле cleaned_data.csv.

Графики анализа данных: numerical_distributions.png, correlation_matrix.png.

Логи обучения моделей (например, gradientbc.log, catboost.log и т.д.).

Графики сравнения моделей: 

model_comparison_accuracy.png,

model_comparison_precision.png,

model_comparison_recall.png, 

model_comparison_f1-score.png.

Лучшую модель, сохраненную в lightgbm_model.pkl.

График сравнения топ-5 моделей: top_models_f1_comparison.png.

Заключение

Лучшей моделью для прогнозирования намерения покупки оказался LightGBM с F1-score 0.6730. В будущем можно улучшить результаты, добавив feature engineering и оптимизацию гиперпараметров.

Как запустить проект

Установите зависимости:

bash
pip install -r requirements.txt
Запустите скрипты в следующем порядке:

bash

python data_loader.py

python online_shoppers_eda.py

python data_cleaning.py

python data_preprocessing.py

python cl_gradientbc.py

python cl_catboost.py

python cl_adaboost.py

python cl_lightGBM.py

python cl_xgboost.py

python model_comparison_viz.py

python final_model.py

