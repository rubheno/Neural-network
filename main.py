import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from skopt import BayesSearchCV

# Cargar los datos.
iris = load_iris()
X = iris.data
y = iris.target

# Dividir los datos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el espacio de búsqueda de hiperparámetros
param_space = {
    'hidden_layer_sizes': [(50,), (100,), (150,), (200,)],
    'activation': ['tanh', 'relu'],
    'alpha': (1e-5, 1e-2, 'log-uniform'),
    'learning_rate_init': (1e-4, 1e-2, 'log-uniform'),
}

# Inicializar el clasificador MLP
mlp = MLPClassifier(max_iter=1000, random_state=42)

# Inicializar la búsqueda bayesiana de hiperparámetros
opt = BayesSearchCV(mlp, param_space, n_iter=30, cv=3, random_state=42)

# Realizar la optimización bayesiana
opt.fit(X_train, y_train)

# Obtener los mejores hiperparámetros encontrados
best_params = opt.best_params_
print("Mejores hiperparámetros encontrados:", best_params)

# Evaluar el modelo con los mejores hiperparámetros en los datos de prueba
accuracy = opt.score(X_test, y_test)
print("Precisión del modelo en datos de prueba:", accuracy)
