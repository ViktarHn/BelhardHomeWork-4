import pandas as pd
import os
import logging

# Настройка логирования
logging.basicConfig(filename='data_loader.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

try:
    if not os.path.exists('online_shoppers.csv'):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
        data = pd.read_csv(url)
        data.to_csv('online_shoppers.csv', index=False)
        logging.info("Данные загружены и сохранены в 'online_shoppers.csv'.")
        print("Данные загружены и сохранены в 'online_shoppers.csv'.")
    else:
        logging.info("Файл 'online_shoppers.csv' уже существует.")
        print("Файл 'online_shoppers.csv' уже существует.")
except Exception as e:
    logging.error(f"Ошибка при загрузке данных: {e}")
    print(f"Ошибка при загрузке данных: {e}")