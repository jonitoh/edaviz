# coding: utf-8
"""Fonctions assez génériques pour être utilisées dans tout le projet."""
import math
import statistics as stat
from enum import Enum

from numpy import datetime64
import numpy as np


SEUIL_CATEGORIE_PAR_DEFAUT = 5


__all__ = [ 'Nature', 'valeur_numpy_nulle', 'donner_type', 'sigmoide', 'inverse_sigmoide' ]


class EnumManager(Enum):
    """Add-ons on Enum objects."""

    @classmethod
    def has_name(cls, name):
        """Check if the name is in the enumeration. """
        return name in cls._member_names_

    @classmethod
    def has_value(cls, value):
        """Check if the value is in the enumeration. """
        return value in cls._value2member_map_

    @classmethod
    def generate_name_from_value(cls, element, has_default_value=True):
        """Ensure coherency in the enumeration """
        # default name
        name = cls._member_names_.keys()[0] if has_default_value else None
        if cls.has_name(element):
            name = element
        elif cls.has_value(element):
            name = cls(element).name
        else:
            pass#print("Beware invalid role value. How is it possible?")
        return name


class Nature(EnumManager):
    """Enumerations of allowed data types."""
    CATEGORY = "Category"
    BINARY = "Binary"
    ORDERED_CATEGORY = "Ordered Category"
    UNORDERED_CATEGORY = "Unordered Category"
    NUMBER = "Number"
    DATETIME = "Datetime"


def valeur_numpy_nulle(variable):
    """ Pour savoir si une variable est NaN, il y a la fonction np.nan.
    Toutefois, elle ne fonctionne que sur des valeurs numériques.
    Cette fonction s'utilise sur tout type de fonction.
        
    Arguments d'entrées:
        variable (Python Object)
        
    Arguments de sorties:
        (bool)
    """
    return variable in [None, ""] or str(variable).lower() == "nan"


def trouver_type(variable, seuil_categorie):
    """Déterminer le type d'une variable par calcul.

    Arguments d'entrées:
        variable (NumPy.array)
        seuil_categorie (int): valeur qui détermine le type d'une variable
            (numérique ou catégorique) suivant le nombre de valeurs distinctes.

    Arguments de sorties:
        (Enum)
    """
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


def donner_type(variable, seuil_categorie=None, type_attitre=None):
    """Retourner le type d'une variable.

    Arguments d'entrées:
        variable (NumPy.array)
        seuil_categorie (int): valeur qui détermine le type d'une variable
            (numérique ou catégorique) suivant le nombre de valeurs distinctes.
        type_attitre (str): possible type donnée par l'utilisateur à vérifier.

    Arguments de sorties:
        (Enum)
    """
    if seuil_categorie is None:
        seuil_categorie = SEUIL_CATEGORIE_PAR_DEFAUT
    
    type_attitre = Nature.generate_name_from_value(type_attitre, has_default_value=False)
    if type_attitre is None:
        return trouver_type(variable, seuil_categorie)
    return type_attitre


def sigmoide(x, valeur_sup=1.0, valeur_inf=.0, pente=1):
    """Personnalisable sigmoide.

    Arguments d'entrées:
        x (float)
        valeur_sup, valeur_inf, pente (float): paramètres
    
    Arguments de sorties:
        (float)
    """
    return valeur_inf + ( valeur_sup - valeur_inf ) / ( 1 + math.exp(- pente * x) )


def inverse_sigmoide(x, valeur_sup=1.0, valeur_inf=.0, pente=1):
    """Personnalisable inverse sigmoide.

    Arguments d'entrées:
        x (float)
        valeur_sup, valeur_inf, pente (float): paramètres
    
    Arguments de sorties:
        (float)
    """
    return math.log( ( x - valeur_inf ) / ( valeur_sup - valeur_inf) ) / pente
