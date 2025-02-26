import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

# Настройка логирования
logging.basicConfig(filename='data_preprocessing.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    data = pd.read_csv('cleaned_data.csv')
    logging.info("Данные успешно загружены для предобработки.")

    # Кодирование категориальных признаков
    data_encoded = pd.get_dummies(data, columns=['Month', 'VisitorType', 'Weekend'], drop_first=True)

    # Масштабирование числовых признаков
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

    logging.info("Предобработка данных завершена.")
    print("\nРазмеры выборок:")
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

except Exception as e:
    logging.error(f"Ошибка при предобработке данных: {e}")
    print(f"Ошибка при предобработке данных: {e}")