import pandas as pd
import os
import logging

# Настройка логирования
logging.basicConfig(filename='data_cleaning.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    if not os.path.exists('online_shoppers.csv'):
        raise FileNotFoundError("Файл 'online_shoppers.csv' не найден!")
    
    data = pd.read_csv('online_shoppers.csv')
    logging.info("Данные успешно загружены для очистки.")
    
    # Удаление дубликатов
    initial_rows = data.shape[0]
    data = data.drop_duplicates()
    removed_rows = initial_rows - data.shape[0]
    logging.info(f"Удалено {removed_rows} дубликатов.")
    
    data.to_csv('cleaned_data.csv', index=False)
    logging.info("Данные очищены и сохранены в 'cleaned_data.csv'.")
    print("Данные очищены и сохранены в 'cleaned_data.csv'.")
except Exception as e:
    logging.error(f"Ошибка при очистке данных: {e}")
    print(f"Ошибка при очистке данных: {e}")