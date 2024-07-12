# Préparation du Jeu de Données
# Chargement et exploration des données
import pandas as pd
from sklearn.datasets import load_iris

# Chargement du jeu de données des iris
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Affichage des premières lignes du DataFrame
print(iris_df.head())

# selection des variables explicative
X = iris_df[['sepal length (cm)']]
y = iris_df['petal length (cm)']

# Modélisation - régression linéaire
# Création et entraînement du modèle de régression linéaire

from sklearn.linear_model import LinearRegression

# Création du modèle de régression linéaire
model = LinearRegression()

# Entraînement du modèle
model.fit(X, y)

# Coefficient (pente)
a = model.coef_[0]

# Interception
b = model.intercept_

print(f"Coefficient : {a}")
print(f"Interception : {b}")

# Analyse des coefficients et interprétation

# 3.Expliquez ce que représentent le coefficient et l'interception dans le contexte du jeu de données des iris.
    # tout simplement le coef représente la pente de la droite de régression, et l'interception est l'ordonnée à l'origine.
# 4. Interprétez les résultats obtenus en termes de relation entre la longueur des sépales et la longueur des pétales.
    #on devrais remarquer une pente positive la longeur des petale augmente avec la longeur des sephale
# 5. Que représente le coefficient dans une régression linéaire ?


# 6. equation de la droite 
y = a * X + b

import matplotlib.pyplot as plt

# Prédictions du modèle
y_pred = model.predict(X)

# Tracé des données et de la droite de régression
plt.scatter(X, y, color='green', label='Données réelles')
plt.plot(X, y_pred, color='yellow', label='Droite de régression')
plt.xlabel('Longueur des sépales (cm)')
plt.ylabel('Longueur des pétales (cm)')
plt.legend()
plt.show()
