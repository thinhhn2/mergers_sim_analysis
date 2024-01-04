import yt
import numpy as np
import sys,os

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

def all_refined_region(r_x,r_y,r_z,lr_x,lr_y,lr_z,spacing):
    """
    This function takes the coordinates of the refined and less-refined dark matter particles
    and returns the boundary of the refined region - the region where there are only refined 
    particles and no less-refined particles.
    
    Parameters
    ----------
    r_x, r_y, r_z: list
        The x,y,z coordinates of all refined dark matter particles.

    lr_x, lr_y, lr_z : list
        The x,y,z coordinates of all less-refined dark matter particles.

    spacing : TYPE, optional
        The spacing when we build another layer to the refined region. 

    Returns
    -------
    [min_x, min_y, min_z, max_x, max_y, max_z] : list of float
        The boundary of the refined region - region where there is no less-refined DM particles.

    """
    #Region where refined particles start to occupy
    rmin_x, rmin_y, rmin_z, rmax_x, rmax_y, rmax_z = min(r_x), min(r_y), min(r_z), max(r_x), max(r_y), max(r_z)
    
    #Restrict the number of less-refined particles to only in the initial refined region to speed up to code
    in_refined = (lr_x > rmin_x) & (lr_x < rmax_x) & (lr_y > rmin_y) & (lr_y < rmax_y) & (lr_z > rmin_z) & (lr_z < rmax_z)
    lr_x = lr_x[in_refined]
    lr_y = lr_y[in_refined]
    lr_z = lr_z[in_refined]
    
    #Building-up method: start from a small region without any less-refined particles, then extend this 
    #region gradually until we hit a less-refined particles
    
    #Set up the initial region that doesn't have less-refined particles
    box_center = [(rmin_x+rmax_x)/2,(rmin_y + rmax_y)/2,(rmin_z + rmax_z)/2]
    
    imin_x = box_center[0] - spacing
    imax_x = box_center[0] + spacing
    imin_y = box_center[1] - spacing
    imax_y = box_center[1] + spacing
    imin_z = box_center[2] - spacing
    imax_z = box_center[2] + spacing
    
    #min_x, min_y, min_z, max_x, max_y, max_z will be updated in the while loop as new boundary
    #for the region
    min_x, min_y, min_z, max_x, max_y, max_z = imin_x, imin_y, imin_z, imax_x, imax_y, imax_z
    #Flags to stop the building-up process
    stop_all = 0
    stop_min_x = 0
    stop_min_y = 0
    stop_min_z = 0
    stop_max_x = 0
    stop_max_y = 0
    stop_max_z = 0
    
    #Check on the initial region to make sure it doesn't contain any less-refined particles
    boolean = (lr_x > min_x) & (lr_x < max_x) & (lr_y > min_y) & (lr_y < max_y) & (lr_z > min_z) & (lr_z < max_z)
    if not any(boolean) == True:
        
        #Each stop_all = 6 means that each building-up direction faces a less-refined particle now
        while stop_all < 6:
        
        #This if stop_min_x is to make sure that the code doesn't try building up this way multiple times
            if stop_min_x < 1:
                nmin_x = min_x - spacing
                boolean = (lr_x > nmin_x) & (lr_x < min_x) & (lr_y > min_y) & (lr_y < max_y) & (lr_z > min_z) & (lr_z < max_z)
                #If there is no less-refined particle(s) in the added region, set the new limit boundary limit
                #If there are, stop the building-up in this direction
                if not any(boolean) == True:
                    min_x = nmin_x
                    #print('min_x updated')
                else:
                    stop_min_x = 1
                    stop_all += 1
                    print('min_x stops being updated')
                    
            if stop_min_y < 1:
                nmin_y = min_y - spacing
                boolean = (lr_x > min_x) & (lr_x < max_x) & (lr_y > nmin_y) & (lr_y < min_y) & (lr_z > min_z) & (lr_z < max_z)
                #If the added region has less-refined particle(s), stop the building-up in this direction
                if not any(boolean) == True:
                    min_y = nmin_y
                    #print('min_y updated')
                else:
                    stop_min_y = 1
                    stop_all += 1
                    print('min_y stops being updated')
                    
            if stop_min_z < 1:
                nmin_z = min_z - spacing
                boolean = (lr_x > min_x) & (lr_x < max_x) & (lr_y > min_y) & (lr_y < max_y) & (lr_z > nmin_z) & (lr_z < min_z)
                #If the added region has less-refined particle(s), stop the building-up in this direction
                if not any(boolean) == True:
                    min_z = nmin_z
                    #print('min_z updated')
                else:
                    stop_min_z = 1
                    stop_all += 1
                    print('min_z stops being updated')
                    
            if stop_max_x < 1:
                nmax_x = max_x + spacing
                boolean = (lr_x > max_x) & (lr_x < nmax_x) & (lr_y > min_y) & (lr_y < max_y) & (lr_z > min_z) & (lr_z < max_z)
                #If there is no less-refined particle(s) in the added region, set the new limit boundary limit
                #If there are, stop the building-up in this direction
                if not any(boolean) == True:
                    max_x = nmax_x
                    #print('max_x updated')
                else:
                    stop_max_x = 1
                    stop_all += 1
                    print('max_x stops being updated')
                    
            if stop_max_y < 1:
                nmax_y = max_y + spacing
                boolean = (lr_x > min_x) & (lr_x < max_x) & (lr_y > max_y) & (lr_y < nmax_y) & (lr_z > min_z) & (lr_z < max_z)
                #If the added region has less-refined particle(s), stop the building-up in this direction
                if not any(boolean) == True:
                    max_y = nmax_y
                    #print('max_y updated')
                else:
                    stop_max_y = 1
                    stop_all += 1
                    print('max_y stops being updated')
                    
            if stop_max_z < 1:
                nmax_z = max_z + spacing
                boolean = (lr_x > min_x) & (lr_x < max_x) & (lr_y > min_y) & (lr_y < max_y) & (lr_z > max_z) & (lr_z < nmax_z)
                #If the added region has less-refined particle(s), stop the building-up in this direction
                if not any(boolean) == True:
                    max_z = nmax_z
                    #print('max_z updated')
                else:
                    stop_max_z = 1
                    stop_all += 1
                    print('max_z stops being updated')
                    
    return [np.array([min_x,min_y,min_z]),np.array([max_x,max_y,max_z])]


def boundary_search_all_snapshots(code_name, lim_index, directory):
    """
    This function runs the all_refined_region function for all snapshots in the simulation.

    INPUT:
    code_name (string): specify the code name of the simulation (ENZO, GADGET3, etc.)

    lim_index (integer): Obtain the nth-most refined particles' coordinate (if the most refined particle is too restrictive)
        1st-most -> index = 0, 2nd-most -> index = 1, 3rd-most -> index = 2, etc.

    dir: the directory of all the snapshots
    """
    os.chdir(directory)

    gs = np.loadtxt('pfs_manual.dat',dtype=str)

    #Runnning parallel to obtain the refined region for each snapshot
    my_storage = {}

    for sto, snapshot in yt.parallel_objects(gs, nprocs-1, storage = my_storage):
        ds = yt.load(snapshot)
        #Set the spacing to be 1/100000 of the box size
        spacing = (ds.domain_right_edge.to('code_length').v[0] - ds.domain_left_edge.to('code_length').v[0])/100000

        reg = ds.all_data()
        
        if code_name == 'GADGET3' or code_name == 'GEAR':
            #Obtain the less-refined particles' coordinates
            lr_x = reg[('PartType5','particle_position_x')].to('code_length').v
            lr_y = reg[('PartType5','particle_position_y')].to('code_length').v
            lr_z = reg[('PartType5','particle_position_z')].to('code_length').v
            lr_m = reg[('PartType5','Masses')].to('Msun').v
        
        if code_name == 'AREPO' or code_name == 'GIZMO':
            #Obtain the less-refined particles' coordinates
            lr_x = reg[('PartType2','particle_position_x')].to('code_length').v
            lr_y = reg[('PartType2','particle_position_y')].to('code_length').v
            lr_z = reg[('PartType2','particle_position_z')].to('code_length').v
            lr_m = reg[('PartType2','Masses')].to('Msun').v
        
        if code_name == 'RAMSES':
            #Obtain the dark matter particles' coordinates
            lr_x = reg[('DM','particle_position_x')].to('code_length').v
            lr_y = reg[('DM','particle_position_y')].to('code_length').v
            lr_z = reg[('DM','particle_position_z')].to('code_length').v
            lr_m = reg[('DM','particle_mass')].to('Msun').v
        
        if code_name == 'ART':
            #Obtain the dark matter particles' coordinates
            lr_x = reg[('darkmatter','particle_position_x')].to('code_length').v
            lr_y = reg[('darkmatter','particle_position_y')].to('code_length').v
            lr_z = reg[('darkmatter','particle_position_z')].to('code_length').v
            lr_m = reg[('darkmatter','particle_mass')].to('Msun').v
        
        if code_name == 'CHANGA':
            #Obtain the dark matter particles' coordinates
            lr_x = reg[('DarkMatter','particle_position_x')].to('code_length').v
            lr_y = reg[('DarkMatter','particle_position_y')].to('code_length').v
            lr_z = reg[('DarkMatter','particle_position_z')].to('code_length').v
            lr_m = reg[('DarkMatter','particle_mass')].to('Msun').v

        #CHOOSING THE LAYER OF REFINEMENT FOR SPH CODES (most refined and other dark matter particles are put in different arrays)
        if code_name == 'GADGET3' or code_name == 'AREPO' or code_name == 'GIZMO' or code_name == 'GEAR':
            if lim_index == 0:
                if code_name == 'GADGET3' or code_name == 'AREPO' or code_name == 'GIZMO':
                    #Obtain the most refined particles' coordinate
                    r_x = reg[('PartType1','particle_position_x')].to('code_length').v
                    r_y = reg[('PartType1','particle_position_y')].to('code_length').v
                    r_z = reg[('PartType1','particle_position_z')].to('code_length').v
                if code_name == 'GEAR':
                    #Obtain the most refined particles' coordinate
                    r_x = reg[('PartType2','particle_position_x')].to('code_length').v
                    r_y = reg[('PartType2','particle_position_y')].to('code_length').v
                    r_z = reg[('PartType2','particle_position_z')].to('code_length').v

            
            if lim_index > 0:
                #The other nth-most refined particles are taken in the less-refined arrays
                mlim = np.sort(np.array(list(set(lr_m))))[lim_index-1]
                r_x = lr_x[lr_m <= mlim]
                r_y = lr_y[lr_m <= mlim]
                r_z = lr_z[lr_m <= mlim]
                #after obtaining the nth-most refined particles, the rest are less-refined
                lr_x = lr_x[lr_m > mlim]
                lr_y = lr_y[lr_m > mlim]
                lr_z = lr_z[lr_m > mlim]
        
        #CHOOSING THE LAYER OF REFINEMENT FOR AMR CODES (all dark matter particles are put in one array)
        if code_name == 'RAMSES' or code_name == 'ART' or code_name == 'CHANGA':
            #Selecting the layer of refined particles
            mlim = np.sort(np.array(list(set(lr_m))))[lim_index]
            r_x = lr_x[lr_m <= mlim]
            r_y = lr_y[lr_m <= mlim]
            r_z = lr_z[lr_m <= mlim]
            #after obtaining the nth-most refined particles, the rest are less-refined
            lr_x = lr_x[lr_m > mlim]
            lr_y = lr_y[lr_m > mlim]
            lr_z = lr_z[lr_m > mlim]

        
        boundary = all_refined_region(r_x, r_y, r_z, lr_x, lr_y, lr_z,spacing=spacing)
        
        sto.result = {}
        sto.result[0] = snapshot
        sto.result[1] = boundary

    output = {}
    if lim_index == 0:
        output_name = 'refined_boundary_1st.npy'
    if lim_index == 1:
        output_name = 'refined_boundary_2nd.npy'
    if lim_index == 2:
        output_name = 'refined_boundary_3rd.npy'
    if lim_index > 2:
        output_name = 'refined_boundary_'+str(lim_index+1)+'th.npy'

    if yt.is_root():
        for c, vals in sorted(my_storage.items()):
            output[vals[0]] = vals[1]
        np.save(output_name,output)
        
    
#------------------------Main code------------------------#
code_name = sys.argv[1]
lim_index = int(sys.argv[2])
directory = sys.argv[3]
boundary_search_all_snapshots(code_name, lim_index, directory)
