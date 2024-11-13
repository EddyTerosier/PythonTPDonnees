# Prédiction de l'Accessibilité PMR des Toilettes Publiques à Paris

Ce projet utilise un modèle d'arbre de décision pour prédire l'accessibilité PMR des toilettes publiques à Paris en fonction de caractéristiques telles que le type de toilette, l'arrondissement, les horaires, et la présence d'un relais bébé.

## Objectif

L'objectif de ce projet est de prédire si une toilette publique est accessible aux personnes à mobilité réduite (PMR) en utilisant les caractéristiques suivantes :
- **TYPE** : Type de toilette (par exemple, SANISETTE, TOILETTES, URINOIR).
- **ARRONDISSEMENT** : Arrondissement de Paris où se situe la toilette.
- **HORAIRE** : Indique si la toilette est ouverte 24 h / 24 ou non.
- **RELAIS_BEBE** : Indique si la toilette possède un relais bébé.

## Données

Les données utilisées pour entraîner l'arbre de décision proviennent d'un fichier CSV (`sanisettesparis.csv`) contenant les informations sur les toilettes publiques à Paris.

Les principales colonnes du dataset sont :
- `TYPE` : Type de la toilette.
- `ARRONDISSEMENT` : Arrondissement de Paris.
- `HORAIRE` : Plage horaire d'ouverture (ex : 24 h / 24).
- `RELAIS_BEBE` : Présence d’un relais bébé.
- `ACCES_PMR` : Accessibilité aux personnes à mobilité réduite, cible de notre prédiction (Oui ou Non).

## Code Explication

Le code effectue les étapes suivantes pour créer et visualiser un modèle d’arbre de décision :

1. **Chargement des Données** :
   ```python
   df = pd.read_csv('sanisettesparis.csv', sep=';')
   # Prédiction de l'Accessibilité PMR des Toilettes Publiques à Paris
    ```

## Étapes du Projet

### Chargement des Données
Charge les données depuis le fichier CSV avec les caractéristiques des toilettes.

### Sélection et Préparation des Données
- Seules les colonnes pertinentes (**TYPE**, **ARRONDISSEMENT**, **HORAIRE**, **RELAIS_BEBE**, **ACCES_PMR**) sont conservées.
- Les lignes avec des valeurs manquantes sont supprimées.

### Encodage des Données
- **TYPE** est transformé en valeurs numériques (chaque type de toilette reçoit un code unique).
- **HORAIRE** est encodé en binaire : `1` pour "24 h / 24" et `0` sinon.
- **RELAIS_BEBE** et **ACCES_PMR** sont également encodés en binaire : `1` pour "Oui" et `0` pour "Non".

### Définition des Caractéristiques et de la Cible
- **X** contient les caractéristiques utilisées pour la prédiction (**TYPE**, **ARRONDISSEMENT**, **HORAIRE**, **RELAIS_BEBE**).
- **y** contient la cible **ACCES_PMR**, indiquant si la toilette est accessible PMR.

### Entraînement de l'Arbre de Décision
- Un modèle d’arbre de décision avec une profondeur maximale de 4 est créé pour éviter le surajustement tout en maintenant une interprétation claire.
- L'arbre est entraîné pour apprendre les relations entre les caractéristiques (**X**) et l'accessibilité PMR (**y**).

### Visualisation de l’Arbre de Décision
- L'arbre est affiché avec `plot_tree`, ce qui permet de visualiser les décisions prises pour prédire l'accessibilité PMR.

## Exemple de Code

Voici un extrait du code d'entraînement et de visualisation de l'arbre :

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Créer et entraîner l'arbre de décision
clf = DecisionTreeClassifier(max_depth=4, random_state=0)
clf.fit(X, y)

# Visualisation de l'arbre de décision
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=['TYPE', 'ARRONDISSEMENT', 'HORAIRE', 'RELAIS_BEBE'],
          class_names=['Non-Accès PMR', 'Accès PMR'])
plt.title("Arbre de Décision pour Prédire l'Accès PMR selon les Types de Toilettes")
plt.show()
```
Interprétation de l'Arbre de Décision
L’arbre de décision utilise les caractéristiques TYPE, ARRONDISSEMENT, HORAIRE, et RELAIS_BEBE pour prédire si une toilette est accessible PMR. À chaque nœud de l’arbre :

Une caractéristique est testée pour diviser les données en sous-groupes.
Chaque sous-groupe est analysé pour voir s’il mène à une prédiction d’accessibilité PMR ou non.
Ce modèle permet d'identifier les types de toilettes et les caractéristiques associées à un accès PMR, ce qui peut aider les usagers à mobilité réduite dans leur recherche de toilettes accessibles.