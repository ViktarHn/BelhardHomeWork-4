import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

# Настройка логирования
logging.basicConfig(filename='adaboost.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка и предобработка данных
try:
    data = pd.read_csv('cleaned_data.csv')
except FileNotFoundError:
    logging.error("Файл 'cleaned_data.csv' не найден! Сначала выполните data_cleaning.py.")
    exit()

data_encoded = pd.get_dummies(data, columns=['Month', 'VisitorType', 'Weekend'], drop_first=True)
numerical_cols = ['Administrative_Duration', 'Informational_Duration', 
                  'ProductRelated_Duration', 'BounceRates', 'ExitRates', 
                  'PageValues', 'SpecialDay']
scaler = StandardScaler()
data_encoded[numerical_cols] = scaler.fit_transform(data_encoded[numerical_cols])

X = data_encoded.drop('Revenue', axis=1)
y = data_encoded['Revenue']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Базовая модель AdaBoost
ab_model = AdaBoostClassifier(random_state=42)
ab_model.fit(X_train, y_train)
y_pred = ab_model.predict(X_test)

# Метрики базовой модели
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

logging.info("AdaBoost (базовые параметры):")
logging.info(f"Accuracy: {accuracy:.4f}")
logging.info(f"Precision: {precision:.4f}")
logging.info(f"Recall: {recall:.4f}")
logging.info(f"F1-score: {f1:.4f}")

# Кросс-валидация для базовой модели
cv_scores = cross_val_score(ab_model, X, y, cv=5, scoring='f1')
logging.info(f"Кросс-валидация (5 фолдов) - Средний F1-score: {cv_scores.mean():.4f}")
logging.info(f"Стандартное отклонение: {cv_scores.std():.4f}")

# Настроенная модель AdaBoost
ab_model_tuned = AdaBoostClassifier(
    random_state=42,
    learning_rate=0.1,  # Уменьшаем скорость обучения
    n_estimators=200    # Больше слабых классификаторов
)
ab_model_tuned.fit(X_train, y_train)
y_pred_tuned = ab_model_tuned.predict(X_test)

# Метрики настроенной модели
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
precision_tuned = precision_score(y_test, y_pred_tuned)
recall_tuned = recall_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned)

logging.info("\nAdaBoost (настроенные параметры):")
logging.info(f"Accuracy: {accuracy_tuned:.4f}")
logging.info(f"Precision: {precision_tuned:.4f}")
logging.info(f"Recall: {recall_tuned:.4f}")
logging.info(f"F1-score: {f1_tuned:.4f}")

# Кросс-валидация для настроенной модели
cv_scores_tuned = cross_val_score(ab_model_tuned, X, y, cv=5, scoring='f1')
logging.info(f"Кросс-валидация (5 фолдов) - Средний F1-score: {cv_scores_tuned.mean():.4f}")
logging.info(f"Стандартное отклонение: {cv_scores_tuned.std():.4f}")