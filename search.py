import numpy as np
import pandas as pd
import datasets
from classifier import classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier


df = pd.read_csv('busca.csv', sep=',')
train_ratio = 0.9
split = int(round(train_ratio * len(df), 0))
target = 'comprou'
df_transformed = pd.get_dummies(df, columns=['busca'])
features = ['home', 'logado', 'busca_algoritmos', 'busca_java', 'busca_ruby']

# define base model
df_train = df[:split]
base_a = df_train[target].sum() / len(df)
base_b = 1 - base_a
base_accuracy = max(base_a, base_b)

# define new model
outcome, accuracy = classifier(
  data=df_transformed, 
  model=AdaBoostClassifier(), 
  features=features, 
  target=target, 
  split=split
)

# verify new model vs base model
verify = pd.DataFrame({
  'pass': base_accuracy < accuracy,
  'base_model': base_accuracy,
  'new_model': accuracy
}, index=[0])
print(verify)
