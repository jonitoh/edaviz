# coding: utf-8
"""
    Functions permettant de calculer les différentes corrélations
    entre les variables d'un même jeu de données.
    Chaque vecteur de variables est un NumPy.array. 
"""
import math
import time
from enum import Enum

from scipy import stats
from numpy import datetime64
import numpy as np
import pandas as pd


SEUIL_CATEGORY_PAR_DEFAUT = 5

class EnumManager(Enum):
    """Fonctionnalités pour mieux gérer les énumérations."""
    
    @classmethod
    def has_name(cls, name):
        """Check if the name is in the enumeration. """
        return name in cls._member_names_
    
    @classmethod
    def has_value(cls, value):
        """Check if the value is in the enumeration. """
        return value in cls._value2member_map_

    @classmethod
    def generate_name_from_value(cls, element):
        """Ensure coherency in the enumeration """
        # default name
        name = cls._member_names_.keys()[0]
        if cls.has_name(element):
            name = element
        elif cls.has_value(element):
            name = cls(element).name
        else:
            print("invalid role value. How possible?")
        return name


class Nature(EnumManager):
    """Enumerations of allowed data types."""
    BINARY = "Binary"
    CATEGORY = "Category"
    NUMBER = "Number"
    DATETIME = "Datetime"


def donner_type(variable, seuil_categorie=SEUIL_CATEGORY_PAR_DEFAUT):
    """Déterminer le type d'une variable."""
    nombre_de_valeurs_distinctes = len(set(variable))
    
    type_variable_categorie = variable.dtype == 'object'
    type_variable_datetime = variable.dtype == datetime64

    if nombre_de_valeurs_distinctes == 2:
        return Nature.BINARY
    elif nombre_de_valeurs_distinctes < seuil_categorie or type_variable_categorie:
        return Nature.CATEGORY
    elif type_variable_datetime:
        return Nature.DATETIME
    else:
        return Nature.NUMBER


def calculer_covariance(variable_1, variable_2):
    """ Calcul de la covariance empirique entre deux variables quantitatives.
        
     Arguments d'entrée:
        variable_1, variable_2 (numpy.array)
        
    Arguments de sortie:
        (float)
    """
    n = len(variable_1)
    moyenne_variable_1 = np.mean(variable_1)
    moyenne_variable_2 = np.mean(variable_2)
    return np.sum(np.dot(variable_1 - moyenne_variable_1, (variable_2 - moyenne_variable_2).T)) / n


def test_correlation_pearson(variable_1, variable_2, covariance=None):
    """Test de corrélation de Pearson
    pour une corrélation entre deux variables quantitatives.
        
    Arguments d'entrées:
        variable_1, variable_2 (NumPy.array)
        covariance (Python function): function pour calculer une covariance entre deux variables.

    Arguments de sorties:
        statistique (float)
        p_value (float)
        coefficient (float)
    """
    if covariance is None:
        covariance = calculer_covariance
    
    # On supprime les valeurs nulles si nécessaire
    condition = np.where(~np.isnan(variable_1) & ~np.isnan(variable_2))
    variable_1, variable_2 = variable_1[condition], variable_2[condition]
    
    n = len(variable_1)

    # Calcul du coefficient
    coefficient = covariance(variable_1, variable_2) / np.sqrt( covariance(variable_1, variable_1) * covariance(variable_2, variable_2) )
    
    # Calcul de la statistique
    statistique = coefficient / np.sqrt((1 - coefficient ** 2)/(n - 2)) if coefficient != 1.0 else np.inf

    # Calcul de la p_value
    p_value = (1 - stats.t.cdf(x=statistique, df=n - 2))
    
    return statistique, p_value, coefficient


def test_correlation_eta_squared(variable_quantitative, variable_qualitative):
    """Test de corrélation de eta squared
    pour une corrélation entre une variable quantitative et une variable qualitative.
        
    Arguments d'entrées:
        variable_quantitative, variable_qualitative (NumPy.array)

    Arguments de sorties:
        statistique (float)
        p_value (float)
        coefficient (float)
    """
    # On supprime les valeurs nulles si nécessaire
    is_null = lambda variable: variable in [ None, '' ] or str(variable).lower() == 'nan'
    is_null = np.vectorize(is_null)
    variable_qualitative = variable_qualitative.astype(str)
    condition = np.where(~np.isnan(variable_quantitative) & ~np.isnan(is_null(variable_qualitative)))
    variable_quantitative, variable_qualitative = variable_quantitative[condition], variable_qualitative[condition]
    
    n_quantitative = len(variable_quantitative)
    classes = np.unique(variable_qualitative)
    n_qualitative = len(classes)

    # Calcul du coefficient
    moyenne_quantitative = variable_quantitative.mean()
    classes = [ variable_quantitative[ np.where(variable_qualitative == classe )] for classe in classes ]
    classes = [ (len(vecteur), np.mean(vecteur, dtype=np.float64)) for vecteur in classes ]
    coefficient_sct = sum((valeur - moyenne_quantitative) ** 2 for valeur in variable_quantitative)
    coefficient_sce = sum(nombre_classe_i * (moyenne_classe_i - moyenne_quantitative) ** 2 for nombre_classe_i, moyenne_classe_i in classes)

    coefficient = coefficient_sce / coefficient_sct if coefficient_sct != 0 else np.inf
    
    # Calcul de la statistique
    statistique = (coefficient * (n_quantitative - n_qualitative)) / ((1 - coefficient) * (n_qualitative - 1))
    
    # Calcul de la p_value
    p_value = 1 - stats.f.cdf(statistique, n_quantitative - n_qualitative, n_qualitative - 1)
    
    return statistique, p_value, coefficient


def calculer_tableau_contingence(variable_1, variable_2, add_total=False):
    """ Calcul du tableau de contingence entre deux variables qualitatives.
        
        Arguments d'entrées:
            variable_1, variable_2 (numpy.array)
        
        Arguments de sortie:
            tableau_de_contingence (NumPy.array) 
    """
    name_1, name_2 = 'variable_1', 'variable_2'
    name_total_1, name_total_2 = 'total_1', 'total_2'
    tableau_de_contingence = (pd.DataFrame(data={name_1: variable_1, name_2: variable_2})
                            .pivot_table(index=name_1, columns=name_1, aggfunc=len))
        
    if add_total:
        tableau_de_contingence.loc[:, name_total_2] = tableau_de_contingence[name_1].value_counts()
        tableau_de_contingence.loc[name_total_1, :] = tableau_de_contingence[name_2].value_counts()
        tableau_de_contingence.loc[name_total_1, name_total_2] = len(variable_1)
        
    return tableau_de_contingence.fillna(0).values


def test_correlation_chi_squared(variable_1, variable_2, contingence=None):
    """Test de corrélation du chi-deux
    pour une corrélation entre deux variables qualtitatives.
        
    Arguments d'entrées:
        variable_1, variable_2 (NumPy.array)
        contingence (Python function): function pour calculer un tableau de contingence entre deux variables.

    Arguments de sorties:
        statistique (float)
        p_value (float)
        coefficient (float)
    """
    if contingence is None:
        contingence = calculer_tableau_contingence
    
    # On supprime les valeurs nulles si nécessaire
    is_null = lambda variable: variable in [ '', 'nan' ]
    is_null = np.vectorize(is_null)
    variable_1, variable_2 = variable_1.astype(str), variable_2.astype(str)
    condition = np.where(~np.isnan(is_null(variable_1)) & ~np.isnan(is_null(variable_2)))
    variable_1, variable_2 = variable_1[condition], variable_2[condition]

    n = len(variable_1)
    tableau_de_contingence = contingence(variable_1, variable_2)
    total_1, total_2 = pd.Series(variable_1).value_counts(), pd.Series(variable_2).value_counts()
    terme_independance = np.dot(total_1, total_2).T

    # Calcul de la statistique
    statistique = ( (tableau_de_contingence - terme_independance) ** 2 / terme_independance ).sum().sum()
    
    # Calcul du coefficient
    coefficient = np.sqrt(statistique / (n * (min(len(total_1), len(total_2)) - 1)))

    # Calcul de la p_value
    p_value = 1 - stats.chi2.cdf(x=statistique, df=(len(total_1) - 1) * (len(total_2) - 1))
    
    return statistique, p_value, coefficient


def test_de_correlation(variable_1, variable_2, *args, **kwargs):
    """Pour calculer la correlation variables.
    
    Arguments d'entrée:
        variable_1, variable_2 (pandas.Series)

    Arguments de sortie:
        statistique, p_value (float or bool)
    """
    variable_1_type = donner_type(variable=variable_1)
    variable_2_type = donner_type(variable=variable_2)

    if (variable_1_type, variable_2_type) == (Nature.NUMBER, Nature.NUMBER):
        covariance = kwargs.get('covariance', None)
        return test_correlation_pearson(variable_1, variable_2, covariance)
    elif variable_1_type == Nature.NUMBER:
        return test_correlation_eta_squared(variable_1, variable_2)
    elif variable_2_type == Nature.NUMBER:
        return test_correlation_eta_squared(variable_2, variable_1)
    else:
        contingence = kwargs.get('contingence', None)
        return test_correlation_chi_squared(variable_1, variable_2, contingence)


def matrice_de_correlation(tableau, format_compact=True, calculer_autocorrelation=False, *args, **kwargs):
    """Pour calculer la matrice de correlation d'un jeu de données.
    
    Arguments d'entrée:
        tableau (Pandas.DataFrame)
        format_compact (bool): Si Vrai, matrice de triplet sinon trois matrices en sortie

    Arguments de sortie:
        correlation or correlation_statistique, correlation_p_value, correlation_coefficient (NumPy.array)
    """
    nom_de_variables = list(tableau)
    correlation = []
    for variable_1 in nom_de_variables:
        correlation_pour_variable_1 = []
        for variable_2 in nom_de_variables:
            triplet = (None, None, None)
            if calculer_autocorrelation and variable_1 == variable_2:
                triplet = (1, None, 1)
            else:
                triplet = test_de_correlation(tableau[variable_1].values,
                                            tableau[variable_2].values,
                                            *args,
                                            **kwargs)
            correlation_pour_variable_1.append(triplet)
        correlation.append(correlation_pour_variable_1)

    if format_compact:
        correlation = np.array(correlation)
        return correlation
    else:
        correlation_statistique = np.array([ list(map(lambda tupl: tupl[0], ligne)) for ligne in correlation ])
        correlation_p_value = np.array([ list(map(lambda tupl: tupl[1], ligne)) for ligne in correlation ])
        correlation_coefficient = np.array([ list(map(lambda tupl: tupl[2], ligne)) for ligne in correlation ])
        
        return correlation_statistique, correlation_p_value, correlation_coefficient


