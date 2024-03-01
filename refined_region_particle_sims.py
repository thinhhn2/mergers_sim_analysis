import yt
import numpy as np
import sys,os
import glob as glob

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

def all_refined_region(r_pos,lr_pos,spacing):
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
    rmin_x, rmin_y, rmin_z, rmax_x, rmax_y, rmax_z = min(r_pos[:,0]), min(r_pos[:,1]), min(r_pos[:,2]), max(r_pos[:,0]), max(r_pos[:,1]), max(r_pos[:,2])
    rmin, rmax = np.array([rmin_x, rmin_y, rmin_z]), np.array([rmax_x, rmax_y, rmax_z])
    #Restrict the number of less-refined particles to only in the initial refined region to speed up to code
    lr_pos = lr_pos[np.all((lr_pos > rmin) & (lr_pos < rmax),axis=1)]
    
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

    # Create the initial box boundaries
    box = np.array([
        [imin_x, imin_y, imin_z],
        [imax_x, imax_y, imax_z]
    ])

    # Define the six directions as 3D vectors
    directions = np.array([
        [[-spacing, 0, 0],[0,0,0]],  # min_x
        [[0,0,0],[spacing, 0, 0]],   # max_x
        [[0, -spacing, 0],[0,0,0]],  # min_y
        [[0,0,0],[0, spacing, 0]],   # max_y
        [[0, 0, -spacing],[0,0,0]],  # min_z
        [[0,0,0],[0, 0, spacing]]    # max_z
    ])

    # Initialize the stop flags
    stop_flags = np.array([False, False, False, False, False, False])

    # While not all directions are stopped
    while not np.all(stop_flags):
        # For each direction
        for i in range(6):
            # If this direction is not stopped
            if stop_flags[i] == False:
                # Try to expand the box in this direction
                new_box = box + directions[i]
                
                # Check if there are any less-refined particles in the expanded region
                boolean = np.all((lr_pos > new_box[0]) & (lr_pos < new_box[1]), axis=1)
                
                # If there are no less-refined particles, update the box
                if not np.any(boolean):
                    box = new_box
                # Otherwise, stop this direction
                else:
                    stop_flags[i] = True

                    
    return [box[0],box[1]]


def boundary_search_all_snapshots(code_name, lim_index, directory, start_idx):
    """
    This function runs the all_refined_region function for all snapshots in the simulation.

    INPUT:
    code_name (string): specify the code name of the simulation (ENZO, GADGET3, etc.)

    lim_index (integer): Obtain the nth-most refined particles' coordinate (if the most refined particle is too restrictive)
        1st-most -> index = 0, 2nd-most -> index = 1, 3rd-most -> index = 2, etc.

    dir: the directory of all the snapshots
    """
    os.chdir(directory)

    gs_full = np.loadtxt('pfs_manual.dat',dtype=str)
    gs = gs_full[start_idx:]

    #Runnning parallel to obtain the refined region for each snapshot
    my_storage = {}

    for sto, snapshot in yt.parallel_objects(gs, nprocs-1, storage = my_storage):
        ds = yt.load(snapshot)
        #Set the spacing to be 1/100000 of the box size
        spacing = (ds.domain_right_edge.to('code_length').v[0] - ds.domain_left_edge.to('code_length').v[0])/100000

        reg = ds.all_data()

        lr_name_dict = {'GEAR': 'PartType5', 'GADGET3': 'PartType5', 'AREPO': 'PartType2', 'GIZMO': 'PartType2', 'RAMSES': 'DM', 'ART': 'darkmatter', 'CHANGA': 'DarkMatter'}
        lr_pos = reg[(lr_name_dict[code_name],'particle_position')].to('code_length').v
        lr_m = reg[(lr_name_dict[code_name],'particle_mass')].to('Msun').v

        #CHOOSING THE LAYER OF REFINEMENT FOR SPH CODES (most refined and other dark matter particles are put in different arrays)
        if code_name == 'GADGET3' or code_name == 'AREPO' or code_name == 'GIZMO' or code_name == 'GEAR':
            if lim_index == 0:
                if code_name == 'GADGET3' or code_name == 'AREPO' or code_name == 'GIZMO':
                    #Obtain the most refined particles' coordinate
                    r_pos = reg[('PartType1','particle_position')].to('code_length').v
                if code_name == 'GEAR':
                    #Obtain the most refined particles' coordinate
                    r_pos = reg[('PartType2','particle_position')].to('code_length').v

            
            if lim_index > 0:
                #The other nth-most refined particles are taken in the less-refined arrays
                mlim = np.sort(np.array(list(set(lr_m))))[lim_index-1]
                r_pos = lr_pos[lr_m <= mlim]
                #after obtaining the nth-most refined particles, the rest are less-refined
                lr_pos = lr_pos[lr_m > mlim]
        
        #CHOOSING THE LAYER OF REFINEMENT FOR AMR CODES (all dark matter particles are put in one array)
        if code_name == 'RAMSES' or code_name == 'ART' or code_name == 'CHANGA':
            #Selecting the layer of refined particles
            mlim = np.sort(np.array(list(set(lr_m))))[lim_index]
            r_pos = lr_pos[lr_m <= mlim]
            #after obtaining the nth-most refined particles, the rest are less-refined
            lr_pos = lr_pos[lr_m > mlim]
        
        boundary = all_refined_region(r_pos, lr_pos, spacing=spacing)

        idx = list(gs_full).index(snapshot)

        output = {}
        if lim_index == 0:
            output_name = 'refined_region_results/refined_boundary_1st_'+str(idx)+'.npy'
        if lim_index == 1:
            output_name = 'refined_region_results/refined_boundary_2nd_'+str(idx)+'.npy'
        if lim_index == 2:
            output_name = 'refined_region_results/refined_boundary_3rd_'+str(idx)+'.npy'
        if lim_index > 2:
            output_name = 'refined_region_results/refined_boundary_'+str(lim_index+1)+'th_'+str(idx)+'.npy'

        output['snapshot'] = snapshot
        output['boundary'] = boundary

        np.save(output_name,output)
        
    
#------------------------Main code------------------------#
code_name = sys.argv[1]
lim_index = int(sys.argv[2])
directory = sys.argv[3]
start_idx = int(sys.argv[4])

if yt.is_root() and os.path.isdir('./refined_region_results') == False:
    os.mkdir('./refined_region_results')

boundary_search_all_snapshots(code_name, lim_index, directory, start_idx)

if yt.is_root():
    refined_files = glob.glob('refined_region_results/*.npy')
    pfs = np.genfromtxt('pfs_manual.dat',dtype=str)
    output_combined = {}

    if len(refined_files) == len(pfs):
        for file in refined_files:
            file_data = np.load(file,allow_pickle=True).tolist()
            output_combined[file_data['snapshot']] = file_data['boundary']  

        if lim_index == 0:
            output_combined_name = 'refined_boundary_1st.npy'
        if lim_index == 1:
            output_combined_name = 'refined_boundary_2nd.npy'
        if lim_index == 2:
            output_combined_name = 'refined_boundary_3rd.npy'
        if lim_index > 2:
            output_combined_name = 'refined_boundary_'+str(lim_index+1)+'th.npy'
        
        np.save(output_combined_name,output_combined)
