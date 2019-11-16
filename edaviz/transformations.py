# coding: utf-8
"""Package centré sur le preprocessing d'un jeu de données.
Bien que basé sur un projet d'analyse de données spécifique (OpenFoodFacts),
le contenu a pour vocation d'être générique."""
import math
import time

import numpy as np
import pandas as pd


def eliminer_colonne_vide(tableau, taux=0.50, retourner_details=True, chemin_fichier='barh_classement_colonnes_vides.png'):
    """Supprimer les colonnes avec un taux de remplissage inférieur à une référence donnée.
    
    Arguments d'entrée:
        tableau (pandas.DataFrame)
        taux (float)
        retourner_details (bool): si True, renvoie le classement des colonnes
            sur leurs taux de remplissage.
    
    Arguments de sortie:
        tableau_nettoye (pandas.DataFrame)
        classement (pandas.DataFrame)
    """
    classement = pd.DataFrame()
    classement['colonne'] = list(tableau)
    classement['taux_de_remplissage'] = (
        classement['colonne']
        .apply(lambda colonne: 1 - tableau[colonne].isna().mean())
    )
    classement = classement.sort_values('taux_de_remplissage', ascending=False)
    
    colonnes_a_eliminer = classement.loc[classement['taux_de_remplissage']>(1-taux), 'colonne']
    tableau_nettoye = tableau.drop(columns=colonnes_a_eliminer)

    if retourner_details:
        (classement
        .plot(kind='barh')
        .get_figure()
        .savefig(chemin_fichier)
        )
        return tableau_nettoye, classement
    
    return tableau_nettoye


def eliminer_ligne_vide(tableau, taux=None, retourner_details=True):
    """Supprimer les lignes avec un taux de remplissage inférieur à une référence donnée.
    
    Arguments d'entrée:
        tableau (pandas.DataFrame)
        taux (float)
        retourner_details (bool): si True, renvoie le classement des colonnes
            sur leurs taux de remplissage.
    
    Arguments de sorties:
        tableau_nettoye (pandas.DataFrame)
        classement (pandas.DataFrame)
    """
    if not taux:
        taux = 0.50
    taux = len(tableau.columns) * taux // 1

    lignes_a_eliminer = tableau.isna().sum(1)
    tableau_nettoye = tableau[ lignes_a_eliminer>=taux ]

    if retourner_details:
        pass
        # (classement
        # .plot(kind='barh')
        # .get_figure()
        # .savefig(chemin_fichier)
        # )
    return tableau_nettoye


def trouver_valeur_par_défaut(valeur_par_defaut, clef):
    """Permet de personnaliser une valeur par défaut suivant un axe."""
    if isinstance(valeur_par_defaut, dict):
        return valeur_par_defaut.get(clef, None)
    return valeur_par_defaut


def premiere_occurence(vecteur, valeur_par_defaut=None):
    """Wrapper de la méthode first_valid_index d'un Pandas.Series.
    
    Arguments d'entrée:
        vecteur (pandas.Series)
        valeur_par_defaut
    Arguments de sortie:
        (Python Object)
    """
    index = vecteur.first_valid_index()
    if index is None:
        return valeur_par_defaut
    return vecteur[index]


def plus_frequente_occurence(vecteur, valeur_par_defaut=None):
    """Wrapper de la méthode first_valid_index d'un Pandas.Series.
    
    Arguments d'entrée:
        vecteur (pandas.Series)
        valeur_par_defaut
    Arguments de sortie:
        (Python Object)
    """
    return vecteur.value_counts(ascending=False, dropna=False).iloc[0]


def calculer_imputation(tableau, strategie='première occurence', valeur_par_defaut=None):
    """Imputer suivant la strategie choisie.
    
    Arguments d'entrée:
        tableau (pandas.DataFrame)
        strategie (str):
            'première occurence' retourne la première valeur non nulle,
            'la plus fréquente' retourne la valeur la plus fréquente non nulle
    
    Arguments de sortie:
        tableau_nettoye (pandas.DataFrame)
    """
    fonction = None
    if strategie =="première occurence":
        fonction = premiere_occurence
    elif strategie=="la plus fréquente":
        fonction = plus_frequente_occurence
    else:
        raise ValueError(f"l'argument strategie ne peut prendre que les valeurs suivantes: 'première occurence' et 'la plus fréquente'. La valeur donnée ici est {strategie}")
    if tableau.empty:
        raise ValueError("l'argument tableau est vide.")
    tableau_impute = {index: fonction(tableau[index], trouver_valeur_par_défaut(valeur_par_defaut, index)) for index in tableau.columns}
    tableau_impute = pd.Series(tableau_impute)
    return tableau_impute


def eliminer_doublons(tableau, colonnes_ciblees=None, strategie='première occurence'):
    """Retourne un tableau avec des individus uniques
    basés sur les colonnes ciblées.
    
    Arguments d'entrée:
        tableau (pandas.DataFrame)
        colonnes_ciblees (list)
        strategie (str): méthode pour trouver l'unique individu
    
    Arguments de sortie:
        tableau_nettoye (pandas.DataFrame)
    """
    if colonnes_ciblees is None or not isinstance(colonnes_ciblees, list):
        if not isinstance(colonnes_ciblees, list):
            print(f"l'argument colonnes_ciblees n'est pas de type list mais {type(colonnes_ciblees)}")
            return tableau.drop_duplicates()
    methode = lambda tableau: calculer_imputation(tableau, strategie)
    return (
        tableau
        .groupby(colonnes_ciblees)
        .agg(methode)
        .reset_index(drop=True)
    )
