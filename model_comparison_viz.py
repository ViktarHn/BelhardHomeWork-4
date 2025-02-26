import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import logging

# Настройка логирования
logging.basicConfig(filename='model_comparison_viz.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Загрузка данных
    data = pd.read_csv('cleaned_data.csv')
    logging.info("Данные успешно загружены для сравнения моделей.")

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

    # Список моделей
    models = {
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Extra Trees": ExtraTreesClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K Neighbors": KNeighborsClassifier(),
        "SVM Linear": SVC(kernel='linear', random_state=42),
        "Dummy": DummyClassifier(strategy='most_frequent', random_state=42),
        "QDA": QuadraticDiscriminantAnalysis(),
        "LightGBM": lgb.LGBMClassifier(random_state=42),
        "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0)
    }

    # Сохранение результатов
    results = []

    for name, model in models.items():
        # Обучение
        model.fit(X_train, y_train)
        # Предсказание
        y_pred = model.predict(X_test)
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        # Добавляем в список
        results.append([name, accuracy, precision, recall, f1])
        logging.info(f"{name}: Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1-score = {f1:.4f}")

    # Таблица результатов
    results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score'])
    print("\nРезультаты всех моделей:")
    print(results_df)

    # Визуализация всех метрик
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.barh(results_df['Model'], results_df[metric], color='skyblue')
        plt.xlabel(metric)
        plt.title(f'Сравнение моделей по {metric}')
        plt.gca().invert_yaxis()
        plt.savefig(f'model_comparison_{metric.lower()}.png')
        plt.show()
        logging.info(f"График сравнения моделей по {metric} сохранен в 'model_comparison_{metric.lower()}.png'.")

except Exception as e:
    logging.error(f"Ошибка при сравнении моделей: {e}")
    print(f"Ошибка при сравнении моделей: {e}")