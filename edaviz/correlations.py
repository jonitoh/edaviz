# coding: utf-8
"""
    Fonctions permettant de calculer les différentes corrélations
    entre les variables d'un même jeu de données.
    Un vecteur peut être une liste, un Pandas.Series ou NumPy.array. 
"""
import math
import statistics as stat
import operator

from scipy import stats
from numpy import datetime64
import numpy as np
import pandas as pd

from .utils import *


def calculer_covariance(variable_1, variable_2):
    """Calcul de la covariance empirique entre deux variables quantitatives.

    Arguments d'entrée:
        variable_1, variable_2 (numpy.array)

    Arguments de sortie:
        (float)
    """
    n = len(variable_1)
    moyenne_variable_1 = stat.mean(variable_1)
    moyenne_variable_2 = stat.mean(variable_2)
    return 0 if n == 0 else sum(x_i_1 * x_i_2 for x_i_1, x_i_2 in zip(variable_1 - moyenne_variable_1, variable_2 - moyenne_variable_2)) / n


def calculer_correlation_pearson(variable_1, variable_2):
    """Calcul de la correlation entre deux variables quantitatives.
    On retourne donc une valeur entrez 0 et 1.

    Arguments d'entrée:
        variable_1, variable_2 (numpy.array)

    Arguments de sortie:
        (float)
    """
    correlation = calculer_covariance(variable_1, variable_2)
    variance_1 = calculer_covariance(variable_1, variable_1)
    variance_2 = calculer_covariance(variable_2, variable_2)

    if variance_1 == 0 or variance_2 == 0:
        return 0
    else:
        return correlation / (math.sqrt(variance_1) * math.sqrt(variance_2))


def test_correlation_pearson(variable_1, variable_2, covariance=None):
    """Test de corrélation de Pearson
    pour une corrélation entre deux variables quantitatives.

    Arguments d'entrées:
        variable_1, variable_2 (NumPy.array)
        covariance (Python function): function pour calculer une covariance entre deux variables.

    Arguments de sorties:
        statistique, p_valeur, coefficient (float)
    """
    if covariance is None:
        covariance = calculer_covariance

    # On supprime les valeurs nulles si nécessaire
    condition = ~( np.isnan(variable_1) | np.isnan(variable_2) )
    variable_1, variable_2 = variable_1[condition], variable_2[condition]

    n = len(variable_1)

    # Calcul du coefficient
    coefficient = calculer_correlation_pearson(variable_1, variable_2)

    # Calcul de la statistique
    statistique = np.inf
    if coefficient != 1.0:
        statistique = coefficient / math.sqrt((1 - coefficient ** 2)/(n - 2))

    # Calcul de la p_valeur
    p_valeur = (1 - stats.t.cdf(x=statistique, df=n - 2))

    return statistique, p_valeur, coefficient


def test_correlation_eta_squared(variable_quantitative, variable_qualitative):
    """Test de corrélation de eta squared
    pour une corrélation entre une variable quantitative et une variable qualitative.

    Arguments d'entrées:
        variable_quantitative, variable_qualitative (NumPy.array)

    Arguments de sorties:
        statistique, p_valeur, coefficient (float)
    """
    # On supprime les valeurs nulles si nécessaire
    variable_qualitative = variable_qualitative.astype(str)

    condition = ~( np.isnan(variable_quantitative) | np.isnan([ valeur_numpy_nulle(x) for x in variable_qualitative ]) )
    variable_quantitative = variable_quantitative[condition]
    variable_qualitative = variable_qualitative[condition]

    n_quantitative = len(variable_quantitative)
    classes = np.unique(variable_qualitative)
    n_qualitative = len(classes)

    # Calcul du coefficient
    moyenne_quantitative = stat.mean(variable_quantitative)
    classes = [ variable_quantitative[variable_qualitative == cl] for cl in classes ]
    classes = [(len(vecteur), stat.mean(vecteur)) for vecteur in classes]
    coefficient_sct = sum((valeur - moyenne_quantitative)
                          ** 2 for valeur in variable_quantitative)
    coefficient_sce = sum(nombre_classe_i * (moyenne_classe_i - moyenne_quantitative)
                          ** 2 for nombre_classe_i, moyenne_classe_i in classes)

    coefficient = coefficient_sce / coefficient_sct if coefficient_sct != 0 else np.inf

    # Calcul de la statistique
    statistique = (coefficient * (n_quantitative - n_qualitative)
                   ) / ((1 - coefficient) * (n_qualitative - 1))

    # Calcul de la p_valeur
    p_valeur = 1 - stats.f.cdf(statistique, n_quantitative -
                              n_qualitative, n_qualitative - 1)

    return statistique, p_valeur, coefficient


def calculer_tableau_contingence(variable_1, variable_2, nom_1=None, nom_2=None, rajouter_colonne_total=False):
    """ Calcul du tableau de contingence entre deux variables qualitatives.
    Basé sur https://openclassrooms.com/fr/courses/4525266-decrivez-et-nettoyez-votre-jeu-de-donnees/4775616-analysez-deux-variables-qualitatives-avec-le-chi-2


    Arguments d'entrées:
        variable_1, variable_2 (numpy.array)
        nom_1, nom_2 (str)

    Arguments de sortie:
        tableau_de_contingence (NumPy.array) 
    """
    if nom_1 is None:
        nom_1 = 'variable_1'
    
    if nom_2 is None:
        nom_2 = 'variable_2'
    
    
    nom_total_1, nom_total_2 = 'total_1', 'total_2'
    tableau_de_contingence = (pd.DataFrame(data={nom_1: variable_1, nom_2: variable_2})
                              .pivot_table(index=nom_1, columns=nom_2, aggfunc=len))

    if rajouter_colonne_total:
        tableau_de_contingence.loc[:,
                                   nom_total_2] = tableau_de_contingence[nom_1].value_counts()
        tableau_de_contingence.loc[nom_total_1,
                                   :] = tableau_de_contingence[nom_2].value_counts()
        tableau_de_contingence.loc[nom_total_1,
                                   nom_total_2] = len(variable_1)

    return tableau_de_contingence.fillna(0).values


def test_correlation_chi_squared(variable_1, variable_2, nom_1=None, nom_2=None, contingence=None, **kwargs):
    """Test de corrélation du chi-deux
    pour une corrélation entre deux variables qualtitatives.

    Arguments d'entrées:
        variable_1, variable_2 (NumPy.array)
        nom_1, nom_2 (str)
        contingence (Python function): function pour calculer un tableau de contingence entre deux variables.

    Arguments de sorties:
        statistique, p_valeur, coefficient (float): coefficient basé sur le coefficient de Cramer 
    """
    if contingence is None:
        contingence = lambda var_1, var_2: calculer_tableau_contingence(variable_1=var_1,
                                                                        variable_2=var_2,
                                                                        nom_1=nom_1,
                                                                        nom_2=nom_2,
                                                                        rajouter_colonne_total=False)

    # On supprime les valeurs nulles si nécessaire
    variable_1 = variable_1.astype(str)
    variable_2 = variable_2.astype(str)
    condition = ~( np.isnan([ valeur_numpy_nulle(x) for x in variable_1]) | np.isnan([ valeur_numpy_nulle(x) for x in variable_2]) )
    variable_1 = variable_1[condition]
    variable_2 = variable_2[condition]

    n = len(variable_1)
    tableau_de_contingence = contingence(variable_1, variable_2)
    total_1 = pd.Series(variable_1).value_counts()
    total_2 = pd.Series(variable_2).value_counts()

    terme_independance = sum( t_1 * t_2 for t_1, t_2 in zip(total_1, total_2) ) / n

    # Calcul de la statistique
    statistique = ( (tableau_de_contingence - terme_independance) ** 2 / terme_independance ).tolist()
    statistique = sum( sum(ligne) for ligne in statistique )              

    # Calcul du coefficient
    coefficient = math.sqrt( statistique / (n * (min(len(total_1), len(total_2)) - 1)) )

    # Calcul de la p_valeur
    p_valeur = 1 - stats.chi2.cdf(x=statistique, df=(len(total_1) - 1) * (len(total_2) - 1))

    return statistique, p_valeur, coefficient


def test_de_correlation(variable_1, variable_2, nom_1=None, nom_2=None, seuil_categorie=None, *args, **kwargs):
    """Pour calculer la correlation variables.

    Arguments d'entrée:
        variable_1, variable_2 (pandas.Series)
        nom_1, nom_2 (str)

    Arguments de sortie:
        statistique, p_valeur, coefficient (float)
    """
    variable_1_type = donner_type(variable=variable_1, seuil_categorie=seuil_categorie, type_attitre=kwargs.get(nom_1, None))
    variable_2_type = donner_type(variable=variable_2, seuil_categorie=seuil_categorie, type_attitre=kwargs.get(nom_2, None))


    if (variable_1_type, variable_2_type) == (Nature.NUMBER, Nature.NUMBER):
        covariance = kwargs.get('covariance', None)
        return test_correlation_pearson(variable_1, variable_2, covariance)
    elif variable_1_type == Nature.NUMBER:
        return test_correlation_eta_squared(variable_1, variable_2)
    elif variable_2_type == Nature.NUMBER:
        return test_correlation_eta_squared(variable_2, variable_1)
    else:
        contingence = kwargs.get('contingence', None)
        return test_correlation_chi_squared(variable_1=variable_1,
                                            variable_2=variable_2,
                                            nom_1=nom_1,
                                            nom_2=nom_2,
                                            contingence=contingence
                                            ) 


def matrice_de_correlation(tableau, format_compact=True, calculer_autocorrelation=False, retourner_liste=False, *args, **kwargs):
    """Pour calculer la matrice de correlation d'un jeu de données.

    Arguments d'entrée:
        tableau (Pandas.DataFrame)
        format_compact (bool): Si Vrai, matrice de triplet sinon trois matrices en sortie
        calculer_autocorrelation (bool): Si Vrai, calculer l'autocorrelation qui doit s'annuler
            #TO FIX à supprimer, peut-être inutile
        retourner_liste (bool): Si vrai retourner une liste
            #TO FIX à supprimer, peut-être inutile

    Arguments de sortie:
        correlation or correlation_statistique, correlation_p_valeur, correlation_coefficient (NumPy.array or list)
    """
    nom_de_variables = list(tableau)
    correlation = []
    for variable_1 in nom_de_variables:
        correlation_pour_variable_1 = []
        for variable_2 in nom_de_variables:
            #print()
            #print(variable_1, " vs ", variable_2)
            triplet = (np.nan, np.nan, np.nan)
            if not calculer_autocorrelation and variable_1 == variable_2:
                triplet = (1, np.nan, 1)
            else:
                triplet = test_de_correlation(tableau[variable_1].values,
                                              tableau[variable_2].values,
                                              *args,
                                              **kwargs)
            correlation_pour_variable_1.append(triplet)
            #print("***********")
        correlation.append(correlation_pour_variable_1)

    if format_compact:
        if not retourner_liste:
            correlation = np.array(correlation)
        return correlation
    else:
        correlation_statistique = [list(map(operator.itemgetter(0), ligne)) for ligne in correlation]
        correlation_p_valeur = [list(map(operator.itemgetter(1), ligne)) for ligne in correlation]
        correlation_coefficient = [list(map(operator.itemgetter(2), ligne)) for ligne in correlation]

        if not retourner_liste:
            correlation_statistique = np.array(correlation_statistique)
            correlation_p_valeur = np.array(correlation_p_valeur)
            correlation_coefficient = np.array(correlation_coefficient)
        return correlation_statistique, correlation_p_valeur, correlation_coefficient


def normaliser_significativite_booleen(p_valeur, alpha, tolerance, *args, **kwargs):
    """Retourne une valeur booléenne pour qualifier la significativité d'un test statistique.

    Arguments d'entrée:
        p_valeur, alpha, tolerance (float)


    Arguments de sortie:
        significativite (bool)
    """
    significativite = p_valeur >= alpha
    return significativite


def normaliser_significativite_numerique(p_valeur, alpha, tolerance, *args, **kwargs):
    """Retourne une valeur entre 0 et 1 pour quantifier la significativité d'un test statistique.

    Arguments d'entrée:
        p_valeur, alpha, tolerance (float)


    Arguments de sortie:
        significativite (float)
    """
    # Première normalisation entre 0 et 1 
    significativite = max(0, (p_valeur - alpha) / (1 - alpha))

    # Seconde normalisation pour changer la répartition des valeurs normalisées
    significativite = sigmoide(x=significativite,
                               valeur_sup=kwargs.get('valeur_sup',1.0),
                               valeur_inf=kwargs.get('valeur_inf',.0),
                               pente=kwargs.get('pente',1))
    return significativite


def matrice_de_significativite(tableau, alpha, retourner_booleen=False, tolerance=.01, *args, **kwargs):
    """Retourne une matrice où chaque valeur sera comprise entre 0 et 1 ou sera booléenne.

    Arguments d'entrée:
            correlation_p_valeur (Numpy.array or list)
            alpha (float)
            retourner_booleen (bool)
            tolerance (float)


        Arguments de sortie:
            tableau_de_taille (Numpy.array)
    """
    to_array = isinstance(tableau, np.ndarray)
    if to_array:
        tableau = tableau.tolist()
    
    normaliser_significativite = None
    if retourner_booleen:
        normaliser_significativite = lambda s: normaliser_significativite_booleen(p_valeur=s,
                                                                                  alpha=alpha,
                                                                                  tolerance=tolerance,
                                                                                  *args, 
                                                                                  **kwargs)
    else:
        normaliser_kwargs = { 'valeur_sup': 1.0,
                              'valeur_inf': tolerance,
                              'pente': 5
                            }
        normaliser_significativite = lambda s: normaliser_significativite_numerique(p_valeur=s,
                                                                                    alpha=alpha,
                                                                                    tolerance=tolerance,
                                                                                    *args, 
                                                                                    **{**kwargs, **normaliser_kwargs})

    tableau_de_taille = [list(map(normaliser_significativite, ligne)) for ligne in tableau]

    if to_array:
        return np.array(tableau_de_taille)
    return tableau_de_taille
