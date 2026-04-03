#Author: Mènéli Herve Adjole

Context: Ce dataset est le resultat d'une analyse de vins produit dans une même région en Italie mais par 03 differents producteurs.L'analyse mesure **13 constituants chimiques** 
(source:https://www.openml.org/search?type=data&status=active&id=43612)


Wine Prediction MLOps Project
Description

Ce projet fournit un service de prédiction de la qualité des vins basé sur un modèle de machine learning.
Il inclut :

Une API REST pour les prédictions (FastAPI).
Une interface web Streamlit pour tester facilement le modèle.
Un pipeline MLOps simple avec gestion des artefacts et des métriques.

Le projet est conçu pour démontrer les bonnes pratiques de déploiement de modèles ML avec un service public.

Fonctionnalités
Prédiction de la qualité des vins à partir de 13 caractéristiques chimiques.

Deux modes d’entrée :
Saisie manuelle des features via l’interface.
Chargement d’un fichier CSV ou JSON contenant les features.
Retour des prédictions et de la probabilité associée (confiance du modèle).
Interface simple et responsive avec Streamlit.
Documentation et endpoints accessibles via /docs sur l’API FastAPI.

Déploiement
L’API est déployée sur Render :
https://ml-wine-prediction.onrender.com
Endpoints principaux :
/predict : POST → Prédiction des vins
/health : GET → Vérifier que l’API est en ligne
/docs : Swagger UI de la documentation

Frontend
L’interface Streamlit est connectée à l’API.
Lance-la localement avec :
pip install -r requirements.txt
streamlit run app.py

Exemple de requête JSON
{
  "features": [
    {
      "alcohol": 14.23,
      "malic_acid": 1.71,
      "ash": 2.43,
      "alcalinity_of_ash": 15.6,
      "magnesium": 127.0,
      "total_phenols": 2.8,
      "flavanoids": 3.06,
      "nonflavanoid_phenols": 0.28,
      "proanthocyanins": 2.29,
      "color_intensity": 5.64,
      "hue": 1.04,
      "od280/od315_of_diluted_wines": 3.92,
      "proline": 1065.0
    }
  ]
}


Auteur
Mènéli Herve Adjole
LinkedIn : (https://www.linkedin.com/in/herve-adjole/)
GitHub : (https://github.com/morxz-dev)


### Tâche: Classification (multiclasse)

### Anatomie des données: 178 échantillons × 13 features numériques + 1 cible
    nombre de features: 14
    type de variables: numérique
    Taille : < 1KB 

### Défis anticipés:
    #Pas de variable manquantes
    #Déséquilibre léger
    #Dataset propre




ML FLOW

### Expérience dans MLflow: Une expérience est un conteneur qui regroupe plusieurs runs liés à un même problème de machine learning.

### Run: Un run est une exécution complète d’un entraînement avec :
    paramètres
    métriques
    artefacts
    modèle
#Author: Mènéli Herve Adjole

### Différence paramètre / métrique / artefact
    Paramètre : configuration du modèle (ex: n_estimators)
    Métrique : performance mesurée (ex: accuracy)
    Artefact : fichier produit (ex: confusion_matrix.png)


### Mon projet, paramètres :

n_estimators
random_state
type de modèle

Métriques :

  "timestamp": ,
  "train_accuracy": ,
  "val_accuracy": ,
  "test_accuracy": ,
  "train_f1_macro": ,
  "val_f1_macro": ,
  "test_f1_macro": ,
  "hyperparams": {
    "n_estimators": ,
    "random_state": 
  }

### Artefacts :
matrice de confusion
feature_schema.json
modèle sauvegardé
run_info.json


### Paramètres enregistrés
n_estimators
random_state
model_type


### Pourquoi  enregistré ces paramètres ?
influencent directement la performance
permettent la reproductibilité
facilitent la comparaison des modèles


### Métriques retenues
accuracy
f1-score macro


### Pourquoi ces métriiques sont adaptées ?
accuracy → performance globale
f1-score → équilibre précision/rappel, utile en classification multi-classe


### Enrichissement du suivi
Artefact choisi
Matrice de confusion


### Pourquoi utile ?
visualise les erreurs du modèle
identifie les classes mal prédites
complète les métriques globales


### Moment du pipeline
entraînement
prédictions
calcul des métriques


### Modifications
variation de n_estimators (hyperparamètre)


### Pourquoi ces tests ?
analyser l’impact du nombre d’arbres
trouver le meilleur compromis performance/coût


### Métrique utilisée
f1-score (macro) car plus robuste que accurary


### Compromis observé
accuracy peut augmenter
mais f1 peut stagner ou diminuer
compromis performance vs complexité


### Une seule métrique suffit ?
Non
une métrique = vision partielle
besoin de plusieurs indicateurs pour une analyse fiable
equilibre performance / complexité / stabilité


### logique MLOps complète :
reproductibilité
expérimentation contrôlée
sélection de modèle



### Test local vs déploiement
Local : environnement contrôlé (ta machine)
Déploiement : environnement distant (cloud), contraintes réseau, sécurité, OS différent


### Rôle du Dockerfile
Définit l’environnement d’exécution (OS + dépendances + commandes)
Permet reproductibilité dans CI/CD


### Pourquoi ça marche en local mais pas en prod
Chemins absolus
Variables d’environnement absentes
Ports mal configurés
Dépendances manquantes
Différences OS


### Endpoint /health
Vérifie que le service est actif
Utilisé par plateformes cloud pour monitoring et restart


### CI vs CD
CI (Continuous Integration) : tester automatiquement le code
CD (Continuous Delivery/Deployment) : déployer automatiquement

#Author: Mènéli Herve Adjole