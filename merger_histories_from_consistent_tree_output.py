#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:42:47 2023

@author: thinhnguyen
"""

import numpy as np
import ytree



#arbor = ytree.load('halo_comparison_using_consistent_tree_output/tree_0_0_0_shield.dat')
arbor = ytree.load('rockstar_halos/trees/tree_0_0_0.dat')


#Setting the constraints when making the merger histories
x_refined_lower = 0.466796875
x_refined_upper = 0.541015625
y_refined_lower = 0.4735351562
y_refined_upper = 0.5391601563
z_refined_lower = 0.465625 
z_refined_upper = 0.5265625
mass_limit = 1e6 #in Msun unit
#radius_limit = 1e-3 #in code unit (this is the radius limit of the largest halo in the tree)


total_result = {}
 
for tree_index in range(len(arbor)):
#for tree_index in range(100):
    
    tree_result = {}
    
    fulltree = list(arbor[tree_index]['tree'])    
    #We obtain the index of the first halo in the examined tree. This is to set the
    #condition for the while lopp later and is also the starting value of our counter
    index_first_halo = fulltree[0]['Depth_first_ID']
    
    #Set a counter for the while loop. 
    index = index_first_halo
    
    #Since the index represents the Depth_first_ID of the halos, we will run through 
    #all the subtrees until the halo's index matches with the number of halos 
    #in the tree plus the starting index (which means that we reach to the last halo
    #of the tree)
    while index < index_first_halo + len(fulltree):
        #Select the subtree from the top halo. This is the main progenitor lineage of the halo
        #We need "index - index_first_halo" because we slice according to the list index, 
        #which is different from the Depth_first_ID index
        subtree = list(fulltree[index - index_first_halo]['prog']) 
        subtree_raw = fulltree[index - index_first_halo]
        #Calculate the gas_mass_fraction and redshift for all the nodes in the subtree
        subtree_list = {}
        
        #If the main progenitor lineage does not satisfy the constraint, remove the whole trees
        if index == index_first_halo and (sum(subtree_raw['prog','x'].to('unitary') < x_refined_lower) > 0 or sum(subtree_raw['prog','x'].to('unitary') > x_refined_upper) > 0 or sum(subtree_raw['prog','y'].to('unitary') < y_refined_lower) > 0 or sum(subtree_raw['prog','y'].to('unitary') > y_refined_upper) > 0 or sum(subtree_raw['prog','z'].to('unitary') < z_refined_lower) > 0 or sum(subtree_raw['prog','z'].to('unitary') > z_refined_upper) > 0 or sum(subtree_raw['prog','mass'].to('Msun') < mass_limit) > 0):
            break
        
        #Setting the constraints on the tree/halo selection
        #All halos of every branch needs to be larger than 10^6 Msun and 
        #be within the refined region
        if sum(subtree_raw['prog','x'].to('unitary') < x_refined_lower) > 0 or sum(subtree_raw['prog','x'].to('unitary') > x_refined_upper) > 0 or sum(subtree_raw['prog','y'].to('unitary') < y_refined_lower) > 0 or sum(subtree_raw['prog','y'].to('unitary') > y_refined_upper) > 0 or sum(subtree_raw['prog','z'].to('unitary') < z_refined_lower) > 0 or sum(subtree_raw['prog','z'].to('unitary') > z_refined_upper) > 0 or sum(subtree_raw['prog','mass'].to('Msun') < mass_limit) > 0:
            index = subtree[-1]['Depth_first_ID'] + 1
            continue 
        
        for j in range(len(subtree)):
            snapshot_index = subtree[j]['Snap_idx']
            subtree_list[str(snapshot_index)] = {}
            subtree_list[str(snapshot_index)]['id'] = int(subtree[j]['id'])
            subtree_list[str(snapshot_index)]['coor'] = [float(subtree[j]['x'].to('unitary')),float(subtree[j]['y'].to('unitary')),float(subtree[j]['z'].to('unitary'))]
            subtree_list[str(snapshot_index)]['Rvir'] = float(subtree[j]['Rvir'].to('unitary'))
            subtree_list[str(snapshot_index)]['mass'] = float(subtree[j]['mass'])
            subtree_list[str(snapshot_index)]['time'] = float(subtree[j]['time']) 
        
        #If this is the main progenitor tree, assigning the branch name, skip the rest of the loop,
        #and restart the loop
        if index == index_first_halo:
            branch = '{}'.format(tree_index)
            tree_result[branch] = subtree_list
            
            index = subtree[-1]['Depth_first_ID'] + 1
            continue
        
        #Obtain all the keys in the current result dictionary
        result_all_key = list(tree_result.keys())
        
        #Loop through the available (sub)trees
        for key, vals in tree_result.items():
            id_list = []
            #Generate the id list for all the halos in a (sub)tree
            for sub_key, sub_vals in vals.items():
                id_list.append(sub_vals['id'])
            #Check what branch the new subtree belongs to by comparing the descendent ID of the first halo
            #to the list of IDs already in our result dictionary
            if subtree[0]['desc_id'] in id_list:
                #Search to see how many branches of a (sub)tree there already are.
                header = key + '_'
                #If the available tree keys have the prefix like the header, add 1 to the sum
                subfix_counter = sum(1 for item in result_all_key if item.startswith(header) and item.count('_')==header.count('_')) 
                #The new branch (or subtree) will be named by the name of its originating branch + '_' + the next counter number
                branch = key + '_' + str(subfix_counter)
                
        #Adding the branch to the result dictionary
        tree_result[branch] = subtree_list
        
        #branch += 1
        #We obtain the Depth_first_ID of the last halo in the subtree. The Depth_first_ID  
        #of the first halo of the next new subtree will be that + 1. This is due to 
        #how the Depth_first_ID order works
        index = subtree[-1]['Depth_first_ID'] + 1
        
        #Merging the dictionaries from each tree
        total_result = total_result | tree_result

key_name = list(total_result.keys())

mapping = {}  # Initialize an empty mapping

#Re-order the name of the tree_index. Instead of using the arbor index, we re-order it from 0 to 
#whatever the number of trees we have
for item in key_name:
    #Obtain the prefix of the tree key name
    first_number = item.split('_')[0]
    #If the first_number first appears in the loop, add it to the mapping. The mapping is responsible
    #for converting the arbor index to a new-ordered index
    if first_number not in mapping:
        mapping[first_number] = str(len(mapping))

#Replacing the arbor-index keys with modified keys
#The number "1" in the item.replace means that the replacement only happens at the first occurence. For example, "3_3" will be changed to "0_3" instead of "0_0"
modified_key_name = [item.replace(item.split('_')[0], mapping[item.split('_')[0]],1) for item in key_name]

# Create a new dictionary with updated keys
modified_total_result = {}

for key, value in total_result.items():
    new_key = modified_key_name[key_name.index(key)]
    modified_total_result[new_key] = value

np.save('halotrees_Thinh.npy',modified_total_result)
