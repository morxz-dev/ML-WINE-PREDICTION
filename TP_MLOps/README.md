Context: Ce dataset est le resultat d'une analyse de vins produit dans une même région en Italie mais par 03 differents producteurs.L'analyse mesure **13 constituants chimiques** 
(source:https://www.openml.org/search?type=data&status=active&id=43612)

Tâche: Classification (multiclasse)

Anatomie des données: 178 échantillons × 13 features numériques + 1 cible
    nombre de features: 14
    type de variables: numérique
    Taille : < 1KB 

Défis anticipés:
    #Pas de variable manquantes
    #Déséquilibre léger
    #Dataset propre


ML FLOW

Expérience dans MLflow: Une expérience est un conteneur qui regroupe plusieurs runs liés à un même problème de machine learning.

Run: Un run est une exécution complète d’un entraînement avec :
    paramètres
    métriques
    artefacts
    modèle

Différence paramètre / métrique / artefact
    Paramètre : configuration du modèle (ex: n_estimators)
    Métrique : performance mesurée (ex: accuracy)
    Artefact : fichier produit (ex: confusion_matrix.png)

    Mon projet

Paramètres :

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

Artefacts :

matrice de confusion
feature_schema.json
modèle sauvegardé
run_info.json

Paramètres enregistrés
n_estimators
random_state
model_type

Pourquoi importants ?
influencent directement la performance
permettent la reproductibilité
facilitent la comparaison des modèles

Métriques retenues
accuracy
f1-score macro

Pourquoi adaptées ?
accuracy → performance globale
f1-score → équilibre précision/rappel, utile en classification multi-classe

Enrichissement du suivi
Artefact choisi

Matrice de confusion

Pourquoi utile ?
visualise les erreurs du modèle
identifie les classes mal prédites
complète les métriques globales

Moment du pipeline

entraînement
prédictions
calcul des métriques

Donc : phase d’évaluation

21. Vérification MLflow

Oui, l’interface contient :

✔ paramètres
✔ métriques
✔ artefacts (confusion_matrix.png, JSON…)
✔ modèle enregistré
🔹 5. Comparaison des expérimentations
Exemple de 3 runs
Run	Modification
Run 1	n_estimators = 50
Run 2	n_estimators = 100
Run 3	n_estimators = 200
22. Modifications
variation de n_estimators (hyperparamètre)
23. Pourquoi ces tests ?
analyser l’impact du nombre d’arbres
trouver le meilleur compromis performance/coût
24. Meilleur run

celui avec :

meilleure performance sur test
bonne généralisation

(ex: n_estimators = 100 ou 200 selon résultats)

25. Métrique utilisée

f1-score (macro)
(car plus robuste que accuracy)

26. Compromis observé

Oui :

accuracy peut augmenter
mais f1 peut stagner ou diminuer

compromis performance vs complexité

27. Une seule métrique suffit ?

Non

Pourquoi :

une métrique = vision partielle
besoin de plusieurs indicateurs pour une analyse fiable
28. Configuration retenue

👉Exemple :

n_estimators = 100
StandardScaler
RandomForest

bon équilibre performance / complexité / stabilité

 logique MLOps complète :

reproductibilité
expérimentation contrôlée
sélection de modèle



1. Test local vs déploiement

Local : environnement contrôlé (ta machine)
Déploiement : environnement distant (cloud), contraintes réseau, sécurité, OS différent

2. Rôle du Dockerfile

Définit l’environnement d’exécution (OS + dépendances + commandes)
Permet reproductibilité dans CI/CD

3. Pourquoi ça marche en local mais pas en prod

Chemins absolus
Variables d’environnement absentes
Ports mal configurés
Dépendances manquantes
Différences OS

4. Endpoint /health

Vérifie que le service est actif
Utilisé par plateformes cloud pour monitoring et restart

5. CI vs CD

CI (Continuous Integration) : tester automatiquement le code
CD (Continuous Delivery/Deployment) : déployer automatiquement