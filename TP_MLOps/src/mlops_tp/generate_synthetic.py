import pandas as pd
from ydata_synthetic.synthesizers import GaussianCopula

# Charger les données originales
data = pd.read_csv("data/wine.csv")  # ou ton CSV de features

# Créer le synthétiseur
model = GaussianCopula()

# Entraîner sur les données réelles
model.fit(data)

# Générer des données synthétiques
synthetic_data = model.sample(100)  # 100 lignes synthétiques
synthetic_data.to_csv("data/wine_synthetic.csv", index=False)

print(synthetic_data.head())
