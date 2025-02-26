import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
data = pd.read_csv(url)

# Сохранение в CSV
data.to_csv('D:/DataScience/BelhardHomeWork/HW4/online_shoppers.csv', index=False)

# Проверка данных
print("Данные загружены. Пример:")
print(data.head())
print("\nСтолбцы:", data.columns.tolist())