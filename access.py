import numpy as np
import pandas as pd
import datasets
from classifier import classifier
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv('acesso.csv', sep=',')

features = ['acessou_home', 'acessou_como_funciona', 'acessou_contato']
split = int(round(len(df) - (len(df) * 0.3), 0))
target = 'comprou'

outcome, accuracy = classifier(
  data=df, 
  model=MultinomialNB(), 
  features=features, 
  target=target, 
  split=split
)
print(accuracy)
