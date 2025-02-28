import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import joblib
import logging

# Настройка логирования
logging.basicConfig(filename='final_model.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Загрузка данных
    data = pd.read_csv('cleaned_data.csv')
    logging.info("Данные успешно загружены для финальной модели.")

    # Кодирование и масштабирование
    data_encoded = pd.get_dummies(data, columns=['Month', 'VisitorType', 'Weekend'], drop_first=True)
    numerical_cols = ['Administrative_Duration', 'Informational_Duration',
                      'ProductRelated_Duration', 'BounceRates', 'ExitRates',
                      'PageValues', 'SpecialDay']
    scaler = StandardScaler()
    data_encoded[numerical_cols] = scaler.fit_transform(data_encoded[numerical_cols])

    # Разделение данных
    X = data_encoded.drop('Revenue', axis=1)
    y = data_encoded['Revenue']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # Лучшая модель: LightGBM
    lgb_model = lgb.LGBMClassifier(random_state=42)
    lgb_model.fit(X_train, y_train)
    y_pred = lgb_model.predict(X_test)

    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])

    print("Итоговая модель: LightGBM")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Кросс-валидация
    cv_scores = cross_val_score(lgb_model, X, y, cv=5, scoring='f1')
    print(f"\nКросс-валидация (5 фолдов):")
    print(f"Средний F1-score: {cv_scores.mean():.4f}")
    print(f"Стандартное отклонение: {cv_scores.std():.4f}")
    print(f"Все значения F1: {cv_scores}")

    # Визуализация F1-score для топ-5 моделей
    models = ['Gradient Boosting', 'LightGBM', 'CatBoost', 'XGBoost', 'AdaBoost']
    f1_scores = [0.6730, 0.6730, 0.6667, 0.6705, 0.6512]

    plt.figure(figsize=(10, 6))
    plt.barh(models, f1_scores, color='skyblue')
    plt.xlabel('F1-score')
    plt.title('Сравнение лучших моделей по F1-score')
    plt.gca().invert_yaxis()
    for i, v in enumerate(f1_scores):
        plt.text(v + 0.005, i, f'{v:.4f}', va='center')
    plt.savefig('top_models_f1_comparison.png')
    plt.show()
    logging.info("График сравнения топ-5 моделей по F1-score сохранен в 'top_models_f1_comparison.png'.")

    # Сохранение модели
    joblib.dump(lgb_model, 'lightgbm_model.pkl')
    logging.info("Модель LightGBM сохранена в файл 'lightgbm_model.pkl'.")

except Exception as e:
    logging.error(f"Ошибка при создании финальной модели: {e}")
    print(f"Ошибка при создании финальной модели: {e}")