import yt
import numpy as np
from yt.data_objects.particle_filters import add_particle_filter
import matplotlib.pyplot as plt
import sys, os

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

#-------------------------------------------------------------------------------------------
#FUNCTIONS
def distance(coor1,coor2):
    dist = np.sqrt((coor1[0]-coor2[0])**2 + (coor1[1]-coor2[1])**2 + (coor1[2]-coor2[2])**2)
    return dist

def merger_compute(tree_key, data_original, mass_ratio_limit):
    
    #Merging timestep 
    merge_key_list = [element for element in list(data_original.keys()) if element.startswith(tree_key + '_') and element.count('_') == 1]

    #Gas mass fraction plot
    merge_ratio_list = []
    merge_time_list = []
    merge_prog_list = []
    merge_second_list = []
    merge_key_good = []
    merging_index_list = []
    merging_mass_index_list = []

    for merge_key in merge_key_list:

        #Obtain the timestep where the merging happens
        merging_index = list(data_original[merge_key].keys())[-1]
        
        #If a sub-branch ends later than the main progenitor ends, remove that sub-branch
        if int(merging_index) >= int(list(data_original[tree_key].keys())[-1]):
            continue

        #Calculating the merger mass ratio before the two halos overlap
        #To do this, we need to go back to the nearest timestep from the merging
        #where the distance between the two halos' center is larger than 
        #the sum of their radii

        #Set a new variable to represent the Timestep where we evaluate the masses
        #In the beginning, we set it equal to the Merging time index. We will
        #change this index later if the two halos overlap each other
        merging_mass_index = merging_index
        #The first timestep of the sub-tree of the merging halo
        merging_tree_starting_index = list(data_original[merge_key].keys())[0]

        #Obtain the coordinate of the halos (from Kirk's merger tree)
        coor_merging = data_original[merge_key][merging_mass_index]['coor']
        coor_prog = data_original[tree_key][merging_mass_index]['coor']

        #Calculate the distance between two halos
        dist = distance(coor_prog,coor_merging)

        #Obtain the radius of the halos (from Kirk's merger tree)
        radius_merging = data_original[merge_key][merging_mass_index]['Rvir']
        radius_prog = data_original[tree_key][merging_mass_index]['Rvir']

        #If the distance is less than the sum of radii (meaning the halos overlap), we go back one timestep (by
        #subtract the merging_mass_index by 1)
        flag = 0
        while dist < radius_merging + radius_prog:
            merging_mass_index = str(int(merging_mass_index) - 1)

            #if the halos overlap even at the beginning of the subtree (which means
            #we can't go back anymore to satisfy the distance requirement),
            #we exclude this merging subtree out of the analysis.
            #The flag variable is used later to skip the rest of the bigger
            #for loop. The "break" stop the while loop completely
            if int(merging_mass_index) < int(merging_tree_starting_index):
                flag = 1
                break
                
            #Sometimes, a halo is removed from the branch because it violates the constraints (coor, radius, or mass).
            #In this case, there is a gap in the branch (for example, Halo 1 then Halo 3 without Halo 2).
            #Thus, the code cannot read the merging_mass_index. In this happens, we skip the rest of the
            #while loop and continue to subtract 1 to the merging_mass_index
            if merging_mass_index not in list(data_original[merge_key]):
                continue

            coor_merging = data_original[merge_key][merging_mass_index]['coor']
            coor_prog = data_original[tree_key][merging_mass_index]['coor']

            radius_merging = data_original[merge_key][merging_mass_index]['Rvir']
            radius_prog = data_original[tree_key][merging_mass_index]['Rvir']

            dist = distance(coor_prog,coor_merging)

        #Skip the rest of the for loop and begin a new loop for the next subtree
        if flag == 1:
            continue    


        #Obtain the merging time 
        #The beginning of the merger is at the latest time when the two halos are still separated
        #The end of the merger is the time where two halos become one (so the time information is found in the main tree rather than a merging branch)
        merge_time_begin = data_original[merge_key][merging_mass_index]['time']
        merge_time_end = data_original[tree_key][str(int(merging_index) + 1)]['time']

        merge_time_list.append([merge_time_begin,merge_time_end])

        #Obtain the mass of the progenitor halo and the incoming halo when the halos
        #don't overlap yet
        prog_mass = data_original[tree_key][merging_mass_index]['total_mass']
        merge_mass = data_original[merge_key][merging_mass_index]['total_mass']
        
        #Obtain the progenitor and the secondary mass of the merger
        merge_prog_list.append(prog_mass)
        merge_second_list.append(merge_mass)

        #Calculate the merger mass ratio
        merge_ratio = merge_mass/prog_mass
        merge_ratio_list.append(merge_ratio)
        
        #Obtain the key of the selected merging tree 
        merge_key_good.append(merge_key)
        
        #Obtain the merging index and merging_mass_index
        merging_index_list.append(merging_index)
        merging_mass_index_list.append(merging_mass_index)

    merge_ratio_list = np.array(merge_ratio_list)
    merge_time_list = np.array(merge_time_list)
    merge_prog_list = np.array(merge_prog_list)
    merge_second_list = np.array(merge_second_list)
    merging_index_list = np.array(merging_index_list)
    merging_mass_index_list = np.array(merging_mass_index_list)
    merge_key_good = np.array(merge_key_good)

    #Remove all the mergers whose mass ratio is less than 10% for better visualization
    merge_time_list = merge_time_list[merge_ratio_list > mass_ratio_limit]
    merge_prog_list = merge_prog_list[merge_ratio_list > mass_ratio_limit]
    merge_second_list = merge_second_list[merge_ratio_list > mass_ratio_limit]
    merging_index_list = merging_index_list[merge_ratio_list > mass_ratio_limit]
    merging_mass_index_list = merging_mass_index_list[merge_ratio_list > mass_ratio_limit]
    merge_key_good = merge_key_good[merge_ratio_list > mass_ratio_limit]
    merge_ratio_list = merge_ratio_list[merge_ratio_list > mass_ratio_limit]
    
    return merge_time_list, merge_ratio_list, merge_prog_list, merge_second_list ,merge_key_good, merging_mass_index_list, merging_index_list

#-------------------------------------------------------------------------------------------
#LOAD DATA
def stars(pfilter, data):
    filter_stars = np.logical_and(data["all", "particle_type"] == 2, data["all", "particle_mass"].to('Msun') > 1)
    return filter_stars

tree_name = sys.argv[1] #for example 'halotree_Thinh_structure_with_com.npy'
code_name = sys.argv[2] #Select among ENZO, GADGET3, AREPO, GIZMO, GEAR, RAMSES, and ART
start_idx = int(sys.argv[3]) #if we want to start from a specific snapshot of the tree (when restarting, for example)
branch_idx = int(sys.argv[4]) #0 for the main branch (i.e. tree key '0' -> the main galaxy), 1 for the first merger, 2 for the second merger, etc. 

tree = np.load(tree_name,allow_pickle=True).tolist()
pfs = np.loadtxt('pfs_manual.dat',dtype='str')

min_ratio = 0.1
merger_list = merger_compute('0',tree,0.1) #Identify the list of the merging satellites and the merging time
merger_key_list = merger_compute('0',tree,min_ratio)[4]
merger_mass_index_list = merger_compute('0',tree,min_ratio)[5]
merger_index_list = merger_compute('0',tree,min_ratio)[6]

if branch_idx == 0:
    branch_key = '0' 
    snapshot_idx = list(tree[branch_key].keys())[start_idx:]
else:
    branch_key = merger_key_list[-branch_idx]
    snapshot_idx_start = list(tree[branch_key].keys()).index(merger_mass_index_list[-branch_idx])
    snapshot_idx_end = list(tree[branch_key].keys()).index(merger_index_list[-branch_idx])
    snapshot_idx = list(tree[branch_key].keys())[snapshot_idx_start:snapshot_idx_end+1] 
    snapshot_idx = snapshot_idx[start_idx:]

if yt.is_root() and os.path.isdir('./metadata') == False:
    os.mkdir('./metadata')

dir_name = './metadata/branch-' + branch_key
if yt.is_root() and os.path.isdir(dir_name) == False:
    os.mkdir(dir_name)
#-------------------------------------------------------------------------------------------
#MAIN CODE

my_storage = {}
for sto, idx in yt.parallel_objects(snapshot_idx, nprocs-1,storage = my_storage):
    if code_name == 'ENZO':
        ds = yt.load(pfs[int(idx)])

        coor = tree[branch_key][idx]['coor']
        rvir = tree[branch_key][idx]['Rvir']

        reg = ds.sphere(coor,(rvir,'code_length'))

        add_particle_filter("stars", function=stars, filtered_type="all", requires=["particle_type","particle_mass"])
        ds.add_particle_filter("stars")

        com_coor_star = reg.quantities.center_of_mass(use_gas = False, use_particles = True, particle_type='stars').to('kpc').v
        com_vel_star = reg.quantities.bulk_velocity(use_gas = False, use_particles = True, particle_type='stars').to('km/s').v

        com_coor_bary = reg.quantities.center_of_mass(use_gas = True, use_particles = True, particle_type='stars').to('kpc').v
        com_vel_bary = reg.quantities.bulk_velocity(use_gas = True, use_particles = True, particle_type='stars').to('km/s').v

        #Calculating stars' metadata
        s_mass_each = reg[("stars", "particle_mass")].in_units("Msun").v.tolist()
        s_coor_each = reg[("stars", "particle_position")].in_units("kpc").v.tolist()
        s_vel_each = reg[("stars", "particle_velocity")].in_units("km/s").v.tolist()

        #Calculating gas' metadata
        g_mass_each = reg[("gas","cell_mass")].in_units("Msun").v.tolist()
        g_x_each = reg[("gas","x")].in_units("kpc").v.tolist()
        g_y_each = reg[("gas","y")].in_units("kpc").v.tolist()
        g_z_each = reg[("gas","z")].in_units("kpc").v.tolist()
        g_velx_each = reg[("gas","velocity_x")].in_units("km/s").v.tolist()
        g_vely_each = reg[("gas","velocity_y")].in_units("km/s").v.tolist()
        g_velz_each = reg[("gas","velocity_z")].in_units("km/s").v.tolist()
        g_coor_each = []
        g_vel_each = []
        for i in range(len(g_x_each)):
            g_coor_each.append([g_x_each[i],g_y_each[i],g_z_each[i]])
            g_vel_each.append([g_velx_each[i],g_vely_each[i],g_velz_each[i]])
        g_coor_each = np.array(g_coor_each)
        g_vel_each = np.array(g_vel_each)

    if code_name == 'GADGET3' or code_name == 'AREPO':
        ds = yt.load(pfs[int(idx)],unit_base = {"length": (1.0, "Mpccm/h")})

        coor = tree['0'][idx]['coor']
        rvir = tree['0'][idx]['Rvir']

        reg = ds.sphere(coor,(rvir,'code_length'))

        com_coor_star = reg.quantities.center_of_mass(use_gas = False, use_particles = True, particle_type='PartType4').to('kpc').v
        com_vel_star = reg.quantities.bulk_velocity(use_gas = False, use_particles = True, particle_type='PartType4').to('km/s').v

        com_coor_bary = reg.quantities.center_of_mass(use_gas = True, use_particles = True, particle_type='PartType4').to('kpc').v
        com_vel_bary = reg.quantities.bulk_velocity(use_gas = True, use_particles = True, particle_type='PartType4').to('km/s').v

        #Calculating stars' metadata
        s_mass_each = reg[("PartType4", "particle_mass")].in_units("Msun").v.tolist()
        s_coor_each = reg[("PartType4", "particle_position")].in_units("kpc").v.tolist()
        s_vel_each = reg[("PartType4", "particle_velocity")].in_units("km/s").v.tolist()

        #Calculating gas' metadata
        g_mass_each = reg[("PartType0", "particle_mass")].in_units("Msun").v.tolist()
        g_coor_each = reg[("PartType0", "particle_position")].in_units("kpc").v.tolist()
        g_vel_each = reg[("PartType0", "particle_velocity")].in_units("km/s").v.tolist()
    
    if code_name == 'GIZMO':
        ds = yt.load(pfs[int(idx)]) #GIZMO automatically includes the correct conversion factor

        coor = tree['0'][idx]['coor']
        rvir = tree['0'][idx]['Rvir']

        reg = ds.sphere(coor,(rvir,'code_length'))

        com_coor_star = reg.quantities.center_of_mass(use_gas = False, use_particles = True, particle_type='PartType4').to('kpc').v
        com_vel_star = reg.quantities.bulk_velocity(use_gas = False, use_particles = True, particle_type='PartType4').to('km/s').v

        com_coor_bary = reg.quantities.center_of_mass(use_gas = True, use_particles = True, particle_type='PartType4').to('kpc').v
        com_vel_bary = reg.quantities.bulk_velocity(use_gas = True, use_particles = True, particle_type='PartType4').to('km/s').v

        #Calculating stars' metadata
        s_mass_each = reg[("PartType4", "particle_mass")].in_units("Msun").v.tolist()
        s_coor_each = reg[("PartType4", "particle_position")].in_units("kpc").v.tolist()
        s_vel_each = reg[("PartType4", "particle_velocity")].in_units("km/s").v.tolist()

        #Calculating gas' metadata
        g_mass_each = reg[("PartType0", "particle_mass")].in_units("Msun").v.tolist()
        g_coor_each = reg[("PartType0", "particle_position")].in_units("kpc").v.tolist()
        g_vel_each = reg[("PartType0", "particle_velocity")].in_units("km/s").v.tolist()

    if code_name == 'GEAR':
        ds = yt.load(pfs[int(idx)]) #GIZMO automatically includes the correct conversion factor

        coor = tree['0'][idx]['coor']
        rvir = tree['0'][idx]['Rvir']

        reg = ds.sphere(coor,(rvir,'code_length'))

        com_coor_star = reg.quantities.center_of_mass(use_gas = False, use_particles = True, particle_type='PartType1').to('kpc').v
        com_vel_star = reg.quantities.bulk_velocity(use_gas = False, use_particles = True, particle_type='PartType1').to('km/s').v

        com_coor_bary = reg.quantities.center_of_mass(use_gas = True, use_particles = True, particle_type='PartType1').to('kpc').v
        com_vel_bary = reg.quantities.bulk_velocity(use_gas = True, use_particles = True, particle_type='PartType1').to('km/s').v

        #Calculating stars' metadata
        s_mass_each = reg[("PartType1", "particle_mass")].in_units("Msun").v.tolist()
        s_coor_each = reg[("PartType1", "particle_position")].in_units("kpc").v.tolist()
        s_vel_each = reg[("PartType1", "particle_velocity")].in_units("km/s").v.tolist()

        #Calculating gas' metadata
        g_mass_each = reg[("PartType0", "particle_mass")].in_units("Msun").v.tolist()
        g_coor_each = reg[("PartType0", "particle_position")].in_units("kpc").v.tolist()
        g_vel_each = reg[("PartType0", "particle_velocity")].in_units("km/s").v.tolist()
    

    if code_name == 'ART':
        ds = yt.load(pfs[int(idx)]) #ART automatically includes the correct conversion factor

        coor = tree['0'][idx]['coor']
        rvir = tree['0'][idx]['Rvir']

        reg = ds.sphere(coor,(rvir,'code_length'))

        com_coor_star = reg.quantities.center_of_mass(use_gas = False, use_particles = True, particle_type='stars').to('kpc').v
        com_vel_star = reg.quantities.bulk_velocity(use_gas = False, use_particles = True, particle_type='stars').to('km/s').v

        com_coor_bary = reg.quantities.center_of_mass(use_gas = True, use_particles = True, particle_type='stars').to('kpc').v
        com_vel_bary = reg.quantities.bulk_velocity(use_gas = True, use_particles = True, particle_type='stars').to('km/s').v

        #Calculating stars' metadata
        s_mass_each = reg[("stars", "particle_mass")].in_units("Msun").v.tolist()
        s_coor_each = reg[("stars", "particle_position")].in_units("kpc").v.tolist()
        s_vel_each = reg[("stars", "particle_velocity")].in_units("km/s").v.tolist()

        #Calculating gas' metadata
        g_mass_each = reg[("gas", "cell_mass")].in_units("Msun").v.tolist()
        g_x_each = reg[("gas","x")].in_units("kpc").v.tolist()
        g_y_each = reg[("gas","y")].in_units("kpc").v.tolist()
        g_z_each = reg[("gas","z")].in_units("kpc").v.tolist()
        g_velx_each = reg[("gas","velocity_x")].in_units("km/s").v.tolist()
        g_vely_each = reg[("gas","velocity_y")].in_units("km/s").v.tolist()
        g_velz_each = reg[("gas","velocity_z")].in_units("km/s").v.tolist()
        g_coor_each = []
        g_vel_each = []
        for i in range(len(g_x_each)):
            g_coor_each.append([g_x_each[i],g_y_each[i],g_z_each[i]])
            g_vel_each.append([g_velx_each[i],g_vely_each[i],g_velz_each[i]])
        g_coor_each = np.array(g_coor_each)
        g_vel_each = np.array(g_vel_each)

    if code_name == 'RAMSES':
        ds = yt.load(pfs[int(idx)]) #RAMSES automatically includes the correct conversion factor

        coor = tree['0'][idx]['coor']
        rvir = tree['0'][idx]['Rvir']

        reg = ds.sphere(coor,(rvir,'code_length'))

        com_coor_star = reg.quantities.center_of_mass(use_gas = False, use_particles = True, particle_type='star').to('kpc').v
        com_vel_star = reg.quantities.bulk_velocity(use_gas = False, use_particles = True, particle_type='star').to('km/s').v

        com_coor_bary = reg.quantities.center_of_mass(use_gas = True, use_particles = True, particle_type='star').to('kpc').v
        com_vel_bary = reg.quantities.bulk_velocity(use_gas = True, use_particles = True, particle_type='star').to('km/s').v

        #Calculating stars' metadata
        s_mass_each = reg[("star", "particle_mass")].in_units("Msun").v.tolist()
        s_coor_each = reg[("star", "particle_position")].in_units("kpc").v.tolist()
        s_vel_each = reg[("star", "particle_velocity")].in_units("km/s").v.tolist()

        #Calculating gas' metadata
        g_mass_each = reg[("gas", "cell_mass")].in_units("Msun").v.tolist()
        g_x_each = reg[("gas","x")].in_units("kpc").v.tolist()
        g_y_each = reg[("gas","y")].in_units("kpc").v.tolist()
        g_z_each = reg[("gas","z")].in_units("kpc").v.tolist()
        g_velx_each = reg[("gas","velocity_x")].in_units("km/s").v.tolist()
        g_vely_each = reg[("gas","velocity_y")].in_units("km/s").v.tolist()
        g_velz_each = reg[("gas","velocity_z")].in_units("km/s").v.tolist()
        g_coor_each = []
        g_vel_each = []
        for i in range(len(g_x_each)):
            g_coor_each.append([g_x_each[i],g_y_each[i],g_z_each[i]])
            g_vel_each.append([g_velx_each[i],g_vely_each[i],g_velz_each[i]])
        g_coor_each = np.array(g_coor_each)
        g_vel_each = np.array(g_vel_each)


    #Calculate the angular momentum of only baryonic matters
    if s_mass_each == []: #if there is no star particles
        bary_coor_each = g_coor_each    
        bary_vel_each = g_vel_each
        bary_mass_each = g_mass_each
    else:
        bary_coor_each = np.concatenate((s_coor_each,g_coor_each),axis=0)
        bary_vel_each = np.concatenate((s_vel_each,g_vel_each),axis=0)
        bary_mass_each = np.concatenate((s_mass_each,g_mass_each),axis=0)
    
    bary_relative_coor_each = bary_coor_each - com_coor_bary
    bary_relative_vel_each = bary_vel_each - com_vel_bary
    bary_relative_momentum_each = []
    for i in range(len(bary_mass_each)):
        bary_relative_momentum_each.append(bary_mass_each[i]*bary_relative_vel_each[i])

    bary_relative_momentum_each = np.array(bary_relative_momentum_each)
    #Calculate the angular momentum
    bary_angmoment_each = np.cross(bary_relative_coor_each,bary_relative_momentum_each)

    output_star = {'com_coor_bary':com_coor_bary,'com_coor_star':com_coor_star,'com_vel_bary':com_vel_bary,'com_vel_star':com_vel_star,'mass':s_mass_each,'coor':s_coor_each,'vel':s_vel_each}
    #output_gas = {'com_coor_bary':com_coor_bary,'com_vel_bary':com_vel_bary,'mass':g_mass_each,'coor':g_coor_each,'vel':g_vel_each}
    output_bary = {'com_coor_bary':com_coor_bary,'com_vel_bary':com_vel_bary,'mass':bary_mass_each,'rel_coor':bary_relative_coor_each,'rel_vel':bary_relative_vel_each,'rel_momentum':bary_relative_momentum_each,'angmoment':bary_angmoment_each}

    np.save('%s/stars_%s.npy' % (dir_name,idx),output_star)
    #np.save('metadata/gas_%s.npy' % idx,output_gas)
    np.save('%s/bary_%s.npy' % (dir_name,idx),output_bary)
