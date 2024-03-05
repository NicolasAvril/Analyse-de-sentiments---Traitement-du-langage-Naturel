README pour le TD d'Analyse de Sentiments utilisant le Traitement Automatique du Langage Naturel (TALN)
Introduction

Ce projet se concentre sur l'application des techniques de TALN pour l'analyse de sentiments dans le cadre du défi SemEval'14 ABSA. Les principaux objectifs comprennent l'analyse de la polarité des mots dans des ensembles de données à l'aide de lexiques de sentiments et le calcul des scores de sentiment basés sur les aspects pour évaluer la performance du modèle sur des données de test étiquetées et non étiquetées.
Objectifs

    Calcul de la Polarité : Calculer la polarité des mots dans deux ensembles de données en utilisant un lexique de sentiment.
    Analyse de Sentiment Basée sur les Aspects : Effectuer une analyse de sentiment basée sur les aspects sur les avis de restaurants et d'ordinateurs portables, en prédisant la polarité du sentiment (positive, négative ou neutre) pour des aspects spécifiques mentionnés dans les textes.

Dépendances

Le projet nécessite Python 3.x et les bibliothèques suivantes :

    spacy : Pour le prétraitement du texte, la tokenisation et la reconnaissance d'entités.
    matplotlib : Pour générer des graphiques de la distribution de polarité.
    nltk : Pour la tokenisation, le marquage de parties du discours et l'analyse de sentiment utilisant SentiWordNet.
    xml.etree.ElementTree : Pour parser les ensembles de données XML.

Assurez-vous d'avoir les dernières versions de ces bibliothèques. Utilisez pip pour installer les packages manquants.
Ensembles de Données

Les ensembles de données sont basés sur les données d'entraînement et de test ABSA SemEval'14 pour les domaines des ordinateurs portables et des restaurants. Chaque ensemble de données contient des fichiers XML avec des avis et des annotations pour différents aspects et leurs polarités de sentiment.
Détails de l'Implémentation
Prétraitement

Le prétraitement du texte implique la tokenisation, le marquage des parties du discours, la reconnaissance des entités nommées (REN) et la gestion de la négation à l'aide de spaCy.
Analyse de la Polarité

L'analyse de la polarité utilise le lexique d'émotion NRC (EmoLex) pour calculer la polarité des mots. L'analyse est réalisée à la fois au niveau des mots et pour des fichiers XML entiers, en agrégeant les comptes de polarité positive et négative.
Analyse de Sentiment Basée sur les Aspects

Cette composante comprend :

    Extraire les aspects et leurs catégories des XML.
    Calculer les scores de sentiment pour les aspects en utilisant SentiWordNet et ajuster les scores en fonction des balises de parties du discours.
    Évaluer la performance du modèle en comparant les polarités de sentiment prédites aux annotations standard d'or.

Métriques d'Évaluation

Les métriques d'évaluation comprennent l'Exactitude, le Rappel et le Score F1, calculés en comparant les sentiments prédits aux données de test étiquetées.
