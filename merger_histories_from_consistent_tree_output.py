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
import yt
from yt.data_objects.particle_filters import add_particle_filter
from functools import reduce

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size
yt.enable_parallelism(communicator=comm)

#Create yt filter for Pop 3 stars
def PopIII(pfilter, data):
     filter_stars_pop3 = np.logical_and(data["all", "particle_type"] == 5, data["all", "particle_mass"].to('Msun') > 1) #Adding pop 3 stars
     return filter_stars_pop3  
 
add_particle_filter("PopIII", function=PopIII, filtered_type="all", requires=["particle_type","particle_mass"])

#Create yt filter for stars in general (Pop 2 + Pop 3)
def stars(pfilter, data):
     filter_stars_type_pop2 = data["all", "particle_type"] == 7
     filter_stars_type_pop3 = np.logical_and(data["all", "particle_type"] == 5, data["all", "particle_mass"].to('Msun') > 1) #Adding pop 3 stars
     filter_stars = np.logical_or(filter_stars_type_pop2, filter_stars_type_pop3) 
     return filter_stars  
    
add_particle_filter("stars", function=stars, filtered_type="all", requires=["particle_type","particle_mass"])



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

def merger_histories(folder, mass_limit = 1e6, radius_limit = 1e-3):
    """

    Parameters
    ----------
    folder : str
        The directory of the folder that contains all the simulation snapshots.
    mass_limit : float, optional
        The minimum mass that a halo can have to be reliable. The default is 1e6.

    Returns
    -------
    total_result : dictionary
        The merger history of the simulation. The halos are put into different merger trees and
        branches, which are represented by the dictionary keys

    """
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
    
    #Counting the number of good trees
    tree_counter = 0
     
    for tree_index in range(len(arbor)):
        
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
            subtree = np.array(list(fulltree[index - index_first_halo]['prog'])) 
            #Save the index of the last halo in the subtree (for looping purpose later)
            subtree_last_index = subtree[-1]['Depth_first_ID']
            subtree_raw = fulltree[index - index_first_halo]
            #Calculate the gas_mass_fraction and redshift for all the nodes in the subtree
            subtree_list = {}
            
            x_lower_bool = subtree_raw['prog','x'].to('unitary') > x_refined_lower
            x_upper_bool = subtree_raw['prog','x'].to('unitary') < x_refined_upper
            y_lower_bool = subtree_raw['prog','y'].to('unitary') > y_refined_lower
            y_upper_bool = subtree_raw['prog','y'].to('unitary') < y_refined_upper
            z_lower_bool = subtree_raw['prog','z'].to('unitary') > z_refined_lower
            z_upper_bool = subtree_raw['prog','z'].to('unitary') < z_refined_upper
            mass_bool = subtree_raw['prog','mass'].to('Msun') > mass_limit
            radius_bool = subtree_raw['prog','Rvir'].to('unitary') > radius_limit
            
            #Combine all the constraint booleans into one array with "and" operator
            all_bool = [x_lower_bool, x_upper_bool, y_lower_bool, y_upper_bool, z_lower_bool, z_upper_bool, mass_bool, radius_bool]
            all_bool_combined = reduce(np.logical_and,all_bool)
            
            #Remove the halos that violate the constraints (coordinate, mass, or radius) out of the subtree
            subtree = subtree[all_bool_combined]
            
            #If all halos in the subtree violate the constraints, skip to the next subtree
            if len(subtree) == 0:
                index = subtree_last_index + 1
                continue
            
            
            for j in range(len(subtree)):
                snapshot_index = subtree[j]['Snap_idx']
                subtree_list[str(snapshot_index)] = {}
                subtree_list[str(snapshot_index)]['id'] = int(subtree[j]['id'])
                subtree_list[str(snapshot_index)]['coor'] = [float(subtree[j]['x'].to('unitary')),float(subtree[j]['y'].to('unitary')),float(subtree[j]['z'].to('unitary'))]
                subtree_list[str(snapshot_index)]['Rvir'] = float(subtree[j]['Rvir'].to('unitary'))
                subtree_list[str(snapshot_index)]['mass'] = float(subtree[j]['mass'])
                subtree_list[str(snapshot_index)]['time'] = float(subtree[j]['time']) 
            
            
            #Obtain all the keys in the current result dictionary
            result_all_key = list(total_result.keys())
            
            #Set up a flag to create a new tree (because after removing some halos, the branch can become disconnected)
            new_tree_counter_flag = 1
            #Loop through the available (sub)trees
            for key, vals in total_result.items():
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
                    #If the branch belongs to other bigger branch, turn off the flaf
                    new_tree_counter_flag = 0
            
            #If the flag is still on, turn that unconnected branch into a tree itself
            if new_tree_counter_flag == 1:
                branch = '{}'.format(tree_counter)
                tree_counter += 1
                    
            #Adding the branch to the result dictionary
            total_result[branch] = subtree_list
            
            #We obtain the Depth_first_ID of the last halo in the subtree. The Depth_first_ID  
            #of the first halo of the next new subtree will be that + 1. This is due to 
            #how the Depth_first_ID order works
            index = subtree_last_index + 1
    
    return total_result

def calculate_properties(halo, sim_data, sfr_avetime = 0.005):
    """
    Parameters
    ----------
    halo: list, the halo information from the dictionary
    sim_data: the loaded simulation
    sfr_avetime: the time to average the SFR. Default = 5 million years (0.005 Gyr)

    Returns
    -------
    A dictionary containing the properties of the input halo
    
    -------
    This function takes the halo information and return the gas and stellar properties
    of a halo

    """
    
    #Load the coordinates and radius of the halo (they are already in code_length unit)
    coor = halo['coor']
    radius = halo['Rvir']
    tree_loc = halo['tree_loc']
    ID = halo['id']
    rvir = halo['Rvir']
    
    
    #Select the region of the halo
    reg = sim_data.sphere(coor,(radius,"code_length")) 
    
    #Calculate the gas mass in the region
    g_mass = np.sum(reg[("gas","cell_mass")]).to('Msun') 
    
    #Obtain the type of the particles to get the dark matter particle mass (type_particle == 1)
    type_particle = reg[('all','particle_type')]
    dm_mass = np.sum(reg[("all","particle_mass")][type_particle == 1].to('Msun'))
    
    #Calculate the H2 mass
    h2_mass = np.sum(reg[("gas","H2_mass")]).to('Msun') 
    
    #Calculate the weighted H2_fraction (the weight is the gas mass)
    h2_fraction_each = reg[("gas","H2_fraction")]
    g_mass_each = reg[("gas","cell_mass")].to('Msun')
    h2_fraction = np.average(h2_fraction_each,weights=g_mass_each)
    
    #Calculate the Pop II star mass
    pop2_mass = np.sum(reg[("all","particle_mass")][type_particle == 7].to('Msun'))
    
    #Create the star-type filter for Pop III stars
    sim_data.add_particle_filter("PopIII")
    
    #Calculate the Pop III star mass
    pop3_mass = np.sum(reg["PopIII", "particle_mass"].in_units("Msun"))
    
    #Create the star-type filter to make it easier to extract the creation time for the SFR calculation
    sim_data.add_particle_filter("stars")
    
    #Calculating total stellar mass
    sm_mass = np.sum(reg["stars", "particle_mass"].in_units("Msun"))
    
    #Get the mass and the formation time for each star particle in the halo
    s_mass_each = reg["stars", "particle_mass"].in_units("Msun")
    formation_time = reg["stars", "creation_time"].in_units("Gyr")
    
    #Averaging the SFR 5 million years before the time of the snapshot
    sf_timescale = sfr_avetime*sim_data.units.Gyr
    currenttime = sim_data.current_time.in_units('Gyr')
    
    sfr = np.sum(s_mass_each[formation_time > currenttime - sf_timescale])/sf_timescale 
    sfr = sfr.in_units('Msun/yr')
    
    #Calculate the mvir total mass from yt
    mvir = np.sum(reg["all", "particle_mass"].in_units("Msun")) + g_mass
    
    #Calculate the gas mass fraction
    g_mass_fraction = g_mass/mvir
    
    #Make a dictionary for the output
    output_dict = {"id":ID,"tree_loc":tree_loc,'coor':coor,'Rvir':rvir,'time':float(currenttime),'gas_mass': float(g_mass), 'gas_mass_frac': float(g_mass_fraction),'dm_mass': float(dm_mass), 'h2_mass': float(h2_mass), 'h2_frac':float(h2_fraction),'pop2_mass': float(pop2_mass), 'pop3_mass': float(pop3_mass), 'star_mass': float(sm_mass), 'sfr': float(sfr), 'total_mass':float(mvir)}
    
    return output_dict

def add_properties(folder, merger_histories):
    """
    This function add the additional properties (mass, SFR, etc.) to the merger history
    file. 

    Parameters
    ----------
    folder : str
        The directory to the folder containing the simulation snaptshots.
    merger_histories : dictionary
        The merger history in the dictionary format.

    Returns
    -------
    halo_by_loc : dictionary
        The merger history accompanied with additional properties of masses and SFR.
        The keys of the dictionary are the tree locations.

    """
    #Obtain the index of the redshifts
    gs = np.loadtxt('%s/pfs_manual.dat' % folder,dtype=str)
    z_index = np.arange(0,len(gs),1)
    
    #Obtains the list of all the tree locations
    tree_loc_list = list(merger_histories.keys())
    
    #Sorting the merger_histories dictionary by redshift instead of tree branches to allow faster parallelization
    halo_by_z = {}
    for i in z_index:
        halos_each = []
        for mainkey, value in merger_histories.items():
            if str(i) in value.keys():
                value_add = value[str(i)]
                value_add['tree_loc'] = mainkey
                halos_each.append(value_add)
        key_name = str(i)
        halo_by_z[key_name] = halos_each
    
    #Runnning parallel to add the halo properties 
    my_storage = {}
    
    for sto, i in yt.parallel_objects(list(halo_by_z.keys()), nprocs-1, storage = my_storage):
        #The key in the halo_ns list is the index of the snapshot in the pfs.dat file (0 corresponds to DD0314, etc.)
        redshift_index = int(i)
        #Each processor obtains all the halos in each snapshot
        all_halos_z = halo_by_z[i]
        #Load the simulation
        sim_data = yt.load('%s/%s' % (folder,gs[redshift_index]))
        #Create an array to store the general halo information
        result_each_z = []
        #Run through the list of halo in one snapshot.
        for j in range(len(all_halos_z)):
            halo = all_halos_z[j]
            result_each_z.append(calculate_properties(halo, sim_data)) #Obtain the halo's properties
        sto.result = {}
        sto.result[0] = i
        sto.result[1] = result_each_z
        
    #Re-arrange the dictionary from redshift-sort to tree-location-sort
    halo_by_loc = {}
    for j in tree_loc_list:
        halos_each_loc = {}
        for c, vals in sorted(my_storage.items()):
            for m in range(len(vals[1])):
                if j == vals[1][m]['tree_loc']:
                    halos_each_loc[vals[0]] = vals[1][m]
        halo_by_loc[j] = halos_each_loc
    
    return halo_by_loc
        
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
result_properties = add_properties(folder, result)

#Save the output
np.save('halotrees_Thinh_testingfunction.npy',result_properties)

