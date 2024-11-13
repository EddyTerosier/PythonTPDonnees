import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Charger le dataset
df = pd.read_csv('sanisettesparis.csv', sep=';')

# 2. Sélectionner les colonnes pertinentes pour la prédiction, en excluant 'HORAIRE'
df = df[['TYPE', 'ARRONDISSEMENT', 'RELAIS_BEBE', 'ACCES_PMR']]

# 3. Nettoyer les données en supprimant les lignes avec des valeurs manquantes
df = df.dropna()

# 4. Encoder les colonnes en valeurs numériques

# Encode 'TYPE' en utilisant un code numérique pour chaque type de toilette
df['TYPE'] = df['TYPE'].astype('category').cat.codes

# Convertir 'RELAIS_BEBE' en binaire (1 pour Oui, 0 pour Non)
df['RELAIS_BEBE'] = df['RELAIS_BEBE'].map({'Oui': 1, 'Non': 0})

# Convertir 'ACCES_PMR' en binaire (1 pour Oui, 0 pour Non)
df['ACCES_PMR'] = df['ACCES_PMR'].map({'Oui': 1, 'Non': 0})

# 5. Séparer les données en caractéristiques et cible
X = df[['TYPE', 'ARRONDISSEMENT', 'RELAIS_BEBE']]
y = df['ACCES_PMR']  # La cible est l'accès PMR

# 6. Créer et entraîner l'arbre de décision sans la colonne 'HORAIRE'
clf = DecisionTreeClassifier(max_depth=4, random_state=0)
clf = clf.fit(X, y)

# 7. Visualiser l'arbre de décision
plt.figure(figsize=(50, 50))
plot_tree(clf, filled=True, feature_names=['TYPE', 'ARRONDISSEMENT', 'RELAIS_BEBE'],
          class_names=['Non-Accès PMR', 'Accès PMR'])
plt.title("Arbre de Décision pour Prédire l'Accès PMR selon les Types de Toilettes (Sans HORAIRE)")
plt.show()

# Donnée de test pour prédire un accès ou non-accès PMR
# Choisir un type, un arrondissement, et un relais bébé qui sont moins susceptibles d'être associés à un accès PMR
data_test = pd.DataFrame({
    'TYPE': [1],           # Exemple d'un autre type encodé
    'ARRONDISSEMENT': [18], # Exemple d'un autre arrondissement
    'RELAIS_BEBE': [0]      # Relais bébé absent
})

# Prédiction avec le modèle entraîné
print("Prédiction pour accès PMR :", clf.predict(data_test)[0])