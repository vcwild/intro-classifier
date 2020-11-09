from sklearn.metrics import accuracy_score


def classifier(data, model, features, target, split):
  X = data[features]
  y = data[target]
  X_train = X[:split]
  y_train = y[:split]
  X_test = X[split:]
  y_test = y[split:]
  model = model
  model.fit(X_train, y_train)
  results = model.predict(X_test)
  accuracy = accuracy_score(y_test, results)

  return [results, accuracy]