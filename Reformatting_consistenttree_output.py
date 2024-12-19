import numpy as np
import ytree
import glob as glob

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#arbor = ytree.load('halo_comparison_using_consistent_tree_output/tree_0_0_0_shield.dat')
arbor = ytree.load('/work/hdd/bbvl/tnguyen2/ENZO/rockstar_halos/trees/arbor/arbor.h5')

#Setting the constraints when making the merger histories
"""
x_refined_lower = 0.466796875
x_refined_upper = 0.541015625
y_refined_lower = 0.4735351562
y_refined_upper = 0.5391601563
z_refined_lower = 0.465625 
z_refined_upper = 0.5265625
mass_limit = 1e6 #in Msun unit
radius_limit = 1e-3 #in code unit
"""
refined_region_files = glob.glob('/work/hdd/bbvl/gtg115x/Halo_Finding/ENZO/refined_region*')
refined_region_data = {}
for file in refined_region_files:
    idx = int((file.split('refined_region_')[1]).split('.npy')[0])
    refined_region_data[idx] = np.load(file, allow_pickle=True)

refined_region_data_keys = np.array(list(refined_region_data.keys()))

total_result = {}
 
my_storage = {}
for sto, tree_index in yt.parallel_objects(range(len(arbor)), nprocs, storage = my_storage):
#for tree_index in range(len(arbor)):
#for tree_index in range(10):
    #
    tree_result = {}
    #
    fulltree = list(arbor[tree_index]['tree'])    
    #We obtain the index of the first halo in the examined tree. This is to set the
    #condition for the while lopp later and is also the starting value of our counter
    index_first_halo = fulltree[0]['Depth_first_ID']
    #
    #Set a counter for the while loop. 
    index = index_first_halo
    #
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
        """
        #If the main progenitor lineage does not satisfy the constraint, remove the whole trees
        if index == index_first_halo and (sum(subtree_raw['prog','x'].to('unitary') < x_refined_lower) > 0 or sum(subtree_raw['prog','x'].to('unitary') > x_refined_upper) > 0 or sum(subtree_raw['prog','y'].to('unitary') < y_refined_lower) > 0 or sum(subtree_raw['prog','y'].to('unitary') > y_refined_upper) > 0 or sum(subtree_raw['prog','z'].to('unitary') < z_refined_lower) > 0 or sum(subtree_raw['prog','z'].to('unitary') > z_refined_upper) > 0 or sum(subtree_raw['prog','mass'].to('Msun') < mass_limit) > 0 or sum(subtree_raw['prog','virial_radius'].to('unitary') < radius_limit) > 0):
            break
        
        #Setting the constraints on the tree/halo selection
        #All halos of every branch needs to be larger than 10^6 Msun and 
        #be within the refined region
        if sum(subtree_raw['prog','x'].to('unitary') < x_refined_lower) > 0 or sum(subtree_raw['prog','x'].to('unitary') > x_refined_upper) > 0 or sum(subtree_raw['prog','y'].to('unitary') < y_refined_lower) > 0 or sum(subtree_raw['prog','y'].to('unitary') > y_refined_upper) > 0 or sum(subtree_raw['prog','z'].to('unitary') < z_refined_lower) > 0 or sum(subtree_raw['prog','z'].to('unitary') > z_refined_upper) > 0 or sum(subtree_raw['prog','mass'].to('Msun') < mass_limit) > 0 or sum(subtree_raw['prog','virial_radius'].to('unitary') < radius_limit) > 0:
            index = subtree[-1]['Depth_first_ID'] + 1
            continue 
        """
        for j in range(len(subtree)):
            snapshot_index = subtree[j]['Snap_idx']
            refined_region = refined_region_data[find_nearest(refined_region_data_keys, snapshot_index)]
            if (subtree[j]['position'].to('unitary').v > refined_region[0]).all() == False or (subtree[j]['position'].to('unitary').v < refined_region[1]).all() == False:
                continue
            subtree_list[snapshot_index] = {}
            subtree_list[snapshot_index]['uid'] = int(subtree[j]['uid'])
            subtree_list[snapshot_index]['Halo_Center'] = subtree[j]['position'].to('unitary').v
            subtree_list[snapshot_index]['Vel_Com'] = subtree[j]['velocity'].to('unitary/s').v
            subtree_list[snapshot_index]['Halo_Radius'] = subtree[j]['virial_radius'].to('unitary').v
            subtree_list[snapshot_index]['Halo_Mass'] = subtree[j]['mass'].to('Msun').v
            subtree_list[snapshot_index]['time'] = subtree[j]['time'].to('Gyr').v
        #
        #If this is the main progenitor tree, assigning the branch name, skip the rest of the loop,
        #and restart the loop
        if index == index_first_halo:
            branch = '{}'.format(tree_index)
            tree_result[branch] = subtree_list
            #
            index = subtree[-1]['Depth_first_ID'] + 1
            continue
        #
        #Obtain all the keys in the current result dictionary
        result_all_key = list(tree_result.keys())
        #
        #Loop through the available (sub)trees
        for key, vals in tree_result.items():
            id_list = []
            #Generate the id list for all the halos in a (sub)tree
            for sub_key, sub_vals in vals.items():
                id_list.append(sub_vals['uid'])
            #Check what branch the new subtree belongs to by comparing the descendent ID of the first halo
            #to the list of IDs already in our result dictionary
            if subtree[0]['desc_uid'] in id_list:
                #Search to see how many branches of a (sub)tree there already are.
                header = key + '_'
                #If the available tree keys have the prefix like the header, add 1 to the sum
                subfix_counter = sum(1 for item in result_all_key if item.startswith(header) and item.count('_')==header.count('_')) 
                #The new branch (or subtree) will be named by the name of its originating branch + '_' + the next counter number
                branch = key + '_' + str(subfix_counter)
        #
        #Adding the branch to the result dictionary
        tree_result[branch] = subtree_list
        #
        #branch += 1
        #We obtain the Depth_first_ID of the last halo in the subtree. The Depth_first_ID  
        #of the first halo of the next new subtree will be that + 1. This is due to 
        #how the Depth_first_ID order works
        index = subtree[-1]['Depth_first_ID'] + 1
        #
        #Merging the dictionaries from each tree
        #total_result = total_result | tree_result
        sto.result = {}
        sto.result[0] = tree_result

output = {}
for c, vals in sorted(my_storage.items()):
    output = output | vals[0]

np.save('/work/hdd/bbvl/tnguyen2/ENZO/halotrees_RCT_reformatted.npy',output)

"""
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
modified_key_name = [item.replace(item.split('_')[0], mapping[item.split('_')[0]]) for item in key_name]

# Create a new dictionary with updated keys
modified_total_result = {}

for key, value in total_result.items():
    new_key = modified_key_name[key_name.index(key)]
    modified_total_result[new_key] = value

np.save('halotrees_Thinh.npy',modified_total_result)
"""

