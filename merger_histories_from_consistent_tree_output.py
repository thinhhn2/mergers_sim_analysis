#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:42:47 2023

@author: thinhnguyen
"""

import numpy as np
import ytree
import sys
import glob as glob
import os


def make_pfs(folder, rockfolder):
    """
    This function creates the pfs.dat file by matching the redshifts in the rockstar
    outputs to the redshifts in the simulation snapshots. This is used when we don't
    know what snapshot each rockstar output refers to because the pfs.dat file is
    missing. This is especially useful if the number of snapshots and the number
    of rockstar outputs are not the same.


    Parameters
    ----------
    folder : str
        The directory to the folder containing the simulation snapshots.
    rockfolder : str
        The name of the folder containing rockstar outputs.

    Returns
    -------
    None.

    """
    cwd = os.getcwd()
    os.chdir('%s/%s' % (folder,rockfolder))

    #Obtaining the directory to all the rockstar outputs
    rockstar_files = glob.glob('out_*')
    rockstar_files = sorted(rockstar_files, key=lambda x: int(x.split('out_')[1].split('.')[0]))

    #Create a dictionary to store the redshifts of all the rockstar outputs and the output's directory
    rockstar_redshifts = {}

    #Loop through each output. The scale factor is the number after the '#a =' string in the
    #output text file
    for file_dir in rockstar_files:
        with open(file_dir, 'r') as file:
            for line in file:
                if line.startswith('#a'):
                    scale = float(line.split()[-1])
                    redshift = 1/scale - 1
                    break
        rockstar_redshifts[file_dir] = redshift

    #Change the directory to where the snapshots are located
    os.chdir('%s' % folder)

    #Obtaining the directory to all the snapshot configuration files
    bases = [["DD", "output_"],
             ["DD", "data"],
             ["DD", "DD"],
             ["RD", "RedshiftOutput"],
             ["RD", "RD"],
             ["RS", "restart"]]
    snapshot_files = []
    for b in bases:
        snapshot_files += glob.glob("%s????/%s????" % (b[0], b[1]))

    #Create a dictionary to store the redshifts of all the snapshots and the snapshot's directory
    snapshot_redshifts = {}

    #Loop through each output. The scale factor is the number after the '#a =' string in the
    #output text file
    for file_dir in snapshot_files:
        with open(file_dir, 'r') as file:
            for line in file:
                if line.startswith('CosmologyCurrentRedshift'):
                    redshift = float(line.split()[-1])
                    break
        snapshot_redshifts[file_dir] = redshift

    #Match the rockstar redshift to closest value in the snapshot redshift to write a pfs.dat file
    pfs_output = []
    for c, vals in rockstar_redshifts.items():
        index = np.argmin(abs(vals - np.array(list(snapshot_redshifts.values()))))
        #Obtain the directory to the snapshot whose redshift is the closest match
        snapshot_dir = list(snapshot_redshifts.keys())[index]
        pfs_output.append(snapshot_dir)

    #Write out a pfs.dat file
    with open('pfs_manual.dat','w') as file:
        for item in pfs_output:
            file.write("%s\n" % item)

    #Change back to the original working directory
    os.chdir(cwd)

def merger_histories(folder, mass_limit = 1e6):
    #arbor = ytree.load('halo_comparison_using_consistent_tree_output/tree_0_0_0_shield.dat')
    arbor = ytree.load('%s/%s/trees/tree_0_0_0.dat' % (folder,rockfolder))
    
    #Setting the refined boundary constraints when making the merger histories
    #Loading a snapshot parameter file to read in the refined boundary region coordinates
    gs = np.loadtxt('%s/pfs_manual.dat' % folder,dtype=str)
    params = open('%s/%s' % (folder,gs[-1]))
    for l in params:
      #Obtain the x,y,z coordinate of the left edge
      if l.startswith('RefineRegionLeftEdge'):
       le = l.split()
       x_refined_lower = float(le[-3])
       y_refined_lower = float(le[-2])
       z_refined_lower = float(le[-1])
      #Obtain the x,y,z coordinate of the right edge
      if l.startswith('RefineRegionRightEdge'):
       re = l.split()
       x_refined_upper = float(re[-3])
       y_refined_upper = float(re[-2])
       z_refined_upper = float(re[-1])
    
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
    
    return modified_total_result

#The directory to the folder containing the simulation snapshots
folder = sys.argv[-1]
rockfolder = 'rockstar_halos'

#If there is no pfs.dat file available, make one, if one exists, copy it.
if os.path.exists('%s/pfs.dat' % folder) == False:
  make_pfs(folder,rockfolder)
else:
  os.system('cp %s/pfs.dat %s/pfs_manual.dat' % (folder,folder))

#Calculate the merger history
result = merger_histories(folder)

#Save the output
np.save('halotrees_Thinh_testingfunction.npy',result)

