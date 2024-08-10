import opendatasets as od
import pandas as pd
from sklearn.model_selection import train_test_split
from fastai.tabular.all import *

# Загрузка датасета с Kaggle
od.download("https://www.kaggle.com/competitions/spaceship-titanic")

# Чтение данных
file_path = 'spaceship-titanic/train.csv'
df = pd.read_csv(file_path)

# Задание 1: Первые 5 строк датасета и типы данных
print(df.head())
print(df.dtypes)

# Задание 2: Предобработка
# Удаление столбца 'PassengerId'
df = df.drop(columns=['PassengerId'])

# Разделение данных на числовые и категориальные
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
cont_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Целевая метрика
dep_var = 'Transported'

# Удаление целевой колонки из признаков
if dep_var in cat_cols:
    cat_cols.remove(dep_var)
if dep_var in cont_cols:
    cont_cols.remove(dep_var)

# Задание 3: Разделение на тренировочные и тестовые данные
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Создание объекта TabularPandas
procs = [Categorify, FillMissing, Normalize]

to = TabularPandas(
    train_df,
    procs=procs,
    cat_names=cat_cols,
    cont_names=cont_cols,
    y_names=dep_var,
    splits=(list(range(len(train_df))), list(range(len(train_df), len(df))))
)

# Загрузка в DataLoader
dls = to.dataloaders(bs=64)

# Задание 4: Создание и обучение модели
learn = tabular_learner(dls, metrics=accuracy)
learn.fit_one_cycle(5, 1e-2)

# Показ результатов
learn.show_results()

# Задание 5: Эксперименты с параметрами
# Попробуем другой learning rate и количество эпох
learn.fit_one_cycle(10, 1e-3)

# Показ результатов с новыми параметрами
learn.show_results()
