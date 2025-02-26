import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Настройка логирования
logging.basicConfig(filename='online_shoppers_eda.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    data = pd.read_csv('online_shoppers.csv')
    logging.info("Данные успешно загружены для EDA.")

    # 1. Структура данных
    print("Столбцы и типы данных:")
    print(data.dtypes)

    # 2. Распределение целевой переменной
    print("\nРаспределение Revenue:")
    print(data['Revenue'].value_counts(normalize=True))

    # 3. Проверка на пропуски и дубликаты
    print("\nПропущенные значения:")
    print(data.isnull().sum())
    print("\nКоличество дубликатов:", data.duplicated().sum())

    # 4. Визуализация распределения числовых признаков
    numerical_cols = ['Administrative', 'Administrative_Duration', 'Informational', 
                      'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 
                      'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numerical_cols, 1):
        plt.subplot(4, 3, i)
        sns.histplot(data[col], kde=True, bins=30)
        plt.title(f'Распределение {col}')
    plt.tight_layout()
    plt.savefig('numerical_distributions.png')
    plt.show()
    logging.info("Графики распределения числовых признаков сохранены в 'numerical_distributions.png'.")

    # 5. Корреляционная матрица
    plt.figure(figsize=(12, 8))
    corr_matrix = data[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Корреляционная матрица числовых признаков')
    plt.savefig('correlation_matrix.png')
    plt.show()
    logging.info("Корреляционная матрица сохранена в 'correlation_matrix.png'.")

except Exception as e:
    logging.error(f"Ошибка при выполнении EDA: {e}")
    print(f"Ошибка при выполнении EDA: {e}")