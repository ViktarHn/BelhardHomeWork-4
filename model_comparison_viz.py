import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import logging
from imblearn.over_sampling import SMOTE
import warnings

# Настройка логирования
logging.basicConfig(filename='model_comparison_viz.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Игнорирование предупреждений
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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

    # Применение SMOTE для балансировки классов
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Разделение данных после балансировки
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.3, stratify=y_resampled, random_state=42
    )

    # Список моделей
    models = {
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=3, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
        "K Neighbors": KNeighborsClassifier(),
        "SVM Linear": SVC(kernel='linear', random_state=42, probability=True),
        "Dummy": DummyClassifier(strategy='most_frequent', random_state=42),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.5),  # Регуляризация для QDA
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, scale_pos_weight=5),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42, eval_metric='logloss'),
        "CatBoost": CatBoostClassifier(iterations=100, depth=3, random_state=42, verbose=0)
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
        precision = precision_score(y_test, y_pred, zero_division=0)  # Учет нулевых предсказаний
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        # Кросс-валидация
        cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=3, scoring='f1')  # Уменьшение фолдов
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        # Добавляем в список
        results.append([name, accuracy, precision, recall, f1, roc_auc, cv_mean, cv_std])
        logging.info(f"{name}: Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1-score = {f1:.4f}, ROC-AUC = {roc_auc:.4f}, CV F1-score = {cv_mean:.4f} ± {cv_std:.4f}")

    # Таблица результатов
    results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'CV F1-score', 'CV Std'])
    print("\nРезультаты всех моделей:")
    print(results_df)

    # Визуализация всех метрик
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'CV F1-score']
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