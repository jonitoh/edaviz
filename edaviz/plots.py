# coding: utf-8
"""Module centré sur le nettoyage et l'analyse exploratoire d'un jeu de données.
Bien que basé sur un projet d'analyse de données spécifique (OpenFoodFacts),
le contenu a pour vocation d'être générale."""
import time

import matplotlib.pyplot as plt
import seaborn as sns



def dessiner_graphique_barres(tableau, colonne, fichier, valeur_manquante='valeur manquante', titre=None, rotation=None):
    """Implémenter un histogramme pour une variable qualitative.
    
    Arguments d'entrée:
        tableau (pandas.DataFrame)
        colonne (str)
        fichier (str): chemin relatif du graphique
        valeur_manquante (str): alias pour les np.nan
        titre (str)
        rotation (int)
    """
    if titre is None:
        titre = f"Histogramme de la variable qualitative {colonne}"
    denombrement = (
        tableau
        [colonne]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .reset_index()
        .rename(columns={'index': colonne, colonne: 'pourcentage'})
        .fillna(valeur_manquante)
    )
    barres = sns.barplot(x=colonne, y="pourcentage", data=denombrement)
    for barre in barres.patches:
        barres.annotate('{:.2f}%'.format(barre.get_height()), (barre.get_x()+0.5, barre.get_height()+1))
    plt.title(titre)
    plt.xlabel(colonne)
    plt.ylabel('pourcentage [%]')
    if rotation:
        plt.setp(barres.get_xticklabels(), rotation=rotation)
    (barres
    .get_figure()
    .savefig(fichier, bbox_inches="tight"))
    plt.close()


def dessiner_distributions_univariees(tableau, nbr_colonnes, fichier, titre=None, dropna=True):
    """Implémenter un histogramme ou une distribution pour chaque variable.
    
    Arguments d'entrée:
        tableau (pandas.DataFrame)
        nbr_colonne (int): on aura une table de graphique de nbr_colonnes colonnes
        fichier (str): chemin relatif du graphique
        titre (str)
    """
    if titre is None:
        titre = f"Distributions univariées"
    nbr_graphiques = len(tableau.columns)
    fig = plt.figure(figsize=(nbr_graphiques, nbr_graphiques))
    for index, colonne in enumerate(tableau.columns):
        vecteur = tableau[colonne]
        if dropna:
            vecteur = vecteur.dropna()
        axe = fig.add_subplot((nbr_graphiques // nbr_colonnes) + 1,
                            nbr_colonnes,
                            index + 1)
        sns.distplot(vecteur,
                    ax=axe,
                    hist=False,
                    #rug=True,
                    kde_kws={"kernel": "gau"}
        )
    plt.axis('tight')
    plt.tight_layout()
    plt.gcf().savefig(fichier, bbox_inches="tight")
    plt.close()


def dessiner_graphique_donut(tableau, colonne, fichier, valeur_manquante='valeur manquante', titre=None):
    """Implémenter un donut pour une variable qualitative.
    
    Arguments d'entrée:
        tableau (pandas.DataFrame)
        colonne (str)
        fichier (str): chemin relatif du graphique
        valeur_manquante (str): alias pour les np.nan
        titre (str)
    """
    if titre is None:
        titre = f"Répartition de la variable qualitative {colonne}"
    camembert = (
        tableau
        [colonne]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .reset_index()
        .rename(columns={'index': colonne, colonne: 'pourcentage'})
        .fillna(valeur_manquante)
    )
    nbr_categories, _ = camembert.shape
    if colormap is None:
        #colormap = plt.get_cmap('tab20')
        colormap = sns.color_palette("husl", nbr_categories)
    # pour isoler la première part
    explode = [0.1] + [0] * (nbr_categories - 1)
    labels = ["{}: {:.2f}%".format(vecteur[colonne], vecteur['pourcentage']) for _, vecteur in camembert.iterrows()]
    plt.pie(camembert['pourcentage'].mul(1/100),
            labels=labels,
            colors=colormap,
            explode=explode,
        )
    trou_du_donut = plt.Circle(xy=(0, 0), radius=0.47, facecolor='white')
    plt.gcf().gca().add_artist(trou_du_donut)
    plt.axis('equal')
    plt.gcf().savefig(fichier, bbox_inches="tight")
    plt.close()



#f, ax = plt.subplots(figsize=(22,5))
#cols = [x for x in df]
#idx = cols.index('nutrition-score-fr_100g')
#sns.heatmap(corr_mat[idx, :].reshape(1, -1),
#            yticklabels='',
#            xticklabels=cols, square=True, linewidths=1, linecolor='white',
#            cbar=True, cbar_kws={"shrink": .45},
#            annot=True, fmt='.3f',
#            cmap="BuPu"
#           );
#plt.title("Corrélation de la variable 'nutrition-score-fr_100g'");
#plt.gcf().savefig('graphs/my_heatmap_score_first.png', bbox_inches="tight")

