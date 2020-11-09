from os import system
from os.path import isfile


if not isfile('./acesso.csv'):
  system('wget "https://s3.amazonaws.com/caelum-online-public/356-intro-machine-learning/files/acesso.csv" --no-check-certificate')

if not isfile('./busca.csv'):
  system('wget "https://s3.amazonaws.com/caelum-online-public/356-intro-machine-learning/busca.csv" --no-check-certificate')