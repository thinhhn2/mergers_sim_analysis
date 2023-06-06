import numpy as np
import itertools

def distance(coor1, coor2):
    """
    This function calculate the distance between two coordinates

    Parameters
    ----------
    coor1 : list
        The coordinate must be in the form of [x,y,z].
    coor2 : list
        The coordinate must be in the form of [x,y,z].

    Returns
    -------
    float
        The distance between two coordinates.

    """
    return np.sqrt((coor1[0] - coor2[0])**2 + (coor1[1] - coor2[1])**2 + (coor1[2] - coor2[2])**2)

def trees_matching(tree1, tree2, distance_limit = 0.25, mass_upper_limit = 1.5, mass_lower_limit = 2/3):
    """
    This function takes the two lists of trees (or branches) and match them with each other based on 
    the mass and coordinates of trees' last halo.

    Parameters
    ----------
    tree1 : dictionary
        This is extracted from the merger history output.
    tree2 : dictionary
        This is extracted from the merger history output.

    Returns
    -------
    match_pair : list
        The list of keys of the matching pairs of trees.

    """
    #Create pairs from the two lists of trees
    pair_list = list(itertools.product(list(tree1),list(tree2)))
    match_pair = []
    for pair in pair_list:
        #Select the last halo in each tree
        halo1 = tree1[pair[0]][list(tree1[pair[0]])[-1]]
        halo2 = tree2[pair[1]][list(tree2[pair[1]])[-1]]
        #Select the matching trees
        if distance(halo1['coor'],halo2['coor'])/halo1['Rvir'] < distance_limit and distance(halo1['coor'],halo2['coor'])/halo2['Rvir'] < distance_limit and halo1['dm_mass']/halo2['dm_mass'] < mass_upper_limit and halo1['dm_mass']/halo2['dm_mass'] > mass_lower_limit:
            match_pair.append(pair)
    
    return match_pair
    