import yt
import numpy as np
import sys,os
import glob as glob
from itertools import product 
from yt.data_objects.unions import ParticleUnion

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

def reduce_range(code_name, directory, ds, ll_all, ur_all, numsegs = 9):
    """
    This function divides a specific domain (given by ll_all and ur_all) into (numsegs-1)^3 segments to speed up the loading process
    and returns the coordinates of the initial refiend region (which will be further refined later)

    Step:
    Go through each segment
    For each segment, record the minimum mass there and the minimum/maximum of the x,y,z coordinates of the minimum-mass particles
    In the main code, based on the lim_index, we will choose what segments to use
    """
    os.chdir(directory)

    #ll_all,ur_all = ds.domain_left_edge, ds.domain_right_edge
    xx,yy,zz = np.meshgrid(np.linspace(ll_all[0],ur_all[0],numsegs),\
                np.linspace(ll_all[1],ur_all[1],numsegs),np.linspace(ll_all[2],ur_all[2],numsegs))
        
    ll = np.concatenate((xx[:-1,:-1,:-1,np.newaxis],yy[:-1,:-1,:-1,np.newaxis],zz[:-1,:-1,:-1,np.newaxis]),axis=3) #ll is lowerleft
    ur = np.concatenate((xx[1:,1:,1:,np.newaxis],yy[1:,1:,1:,np.newaxis],zz[1:,1:,1:,np.newaxis]),axis=3) #ur is upperright

    #ll = np.reshape(ll,(ll.shape[0]**3,3)).v
    #ur = np.reshape(ur,(ur.shape[0]**3,3)).v
    ll = np.reshape(ll,(ll.shape[0]**3,3))
    ur = np.reshape(ur,(ur.shape[0]**3,3))

    _,segdist = np.linspace(ll_all[0],ur_all[0],numsegs,retstep=True)

    #Runnning parallel to obtain the refined region for each snapshot
    my_storage = {}

    for sto, i in yt.parallel_objects(range(len(ll)), nprocs, storage = my_storage):

        #buffer = (segdist/100)*ds.units.code_length #set buffer when loading the box
        #reg = ds.box(ll[i] - buffer ,ur[i] + buffer) 
        reg = ds.box(ll[i],ur[i])

        #Set the spacing to be 1/100000 of the box size
        #spacing = (ds.domain_right_edge.to('code_length').v[0] - ds.domain_left_edge.to('code_length').v[0])/100000

        #combine less-refined particles and refined-particles into one field for GEAR, GIZMO, AREPO, and GADGET3
        if code_name == 'GEAR':
            dm = ParticleUnion("DarkMatter",["PartType5","PartType2"])
            ds.add_particle_union(dm)
        if code_name == 'GADGET3':
            dm = ParticleUnion("DarkMatter",["PartType5","PartType1"])
            ds.add_particle_union(dm)
        if code_name == 'AREPO' or code_name == 'GIZMO':
            dm = ParticleUnion("DarkMatter",["PartType2","PartType1"])
            ds.add_particle_union(dm)

        lr_name_dict = {'GEAR': 'DarkMatter', 'GADGET3': 'DarkMatter', 'AREPO': 'DarkMatter', 'GIZMO': 'DarkMatter', 'RAMSES': 'DM', 'ART': 'darkmatter', 'CHANGA': 'DarkMatter'}
        lr_m = reg[(lr_name_dict[code_name],'particle_mass')].to('Msun').v

        #CHOOSING THE LAYER OF REFINEMENT FOR SPH CODES (most refined and other dark matter particles are put in different arrays)
        if code_name == 'GADGET3' or code_name == 'AREPO' or code_name == 'GIZMO' or code_name == 'GEAR':
            mmin = np.sort(np.array(list(set(lr_m))))[0]
            mtype = len(list(set(lr_m)))
        
        #CHOOSING THE LAYER OF REFINEMENT FOR AMR CODES (all dark matter particles are put in one array)
        if code_name == 'RAMSES' or code_name == 'ART' or code_name == 'CHANGA':
            mmin = np.sort(np.array(list(set(lr_m))))[0]
            mtype = len(list(set(lr_m)))
        
        sto.result = {}
        sto.result[0] = [ll[i],ur[i]]
        sto.result[1] = mmin
        sto.result[2] = mtype
        
    #Re-arrange the dictionary from redshift-sort to tree-location-sort
    seg_pos_list = []
    mlim_list = []
    mtype_list = []
    for c, vals in sorted(my_storage.items()):
        seg_pos_list.append(vals[0])
        mlim_list.append(vals[1])
        mtype_list.append(vals[2])

    return np.array(seg_pos_list), np.array(mlim_list), np.array(mtype_list), segdist
        
def volume_cal(ll,ur):
    return np.product(np.array(ur) - np.array(ll))    

def find_lines(ll_pos,segdist,direction):
    """
    THIS FUNCTION FINDS THE 1D LINE THAT ARE MADE FROM ALL THE SUB-BOXES

    Parameters
    ----------
    ll_pos : TYPE
        LOCATION OF THE LOWER-LEFT CORNER OF THE BOXES.
    segdist : TYPE
        THE SUB-BOX LENGTH.
    direction : TYPE
        FIND THE LINES IN X, Y, OR Z DIRECTION.

    Returns
    -------
    THE COORDINATE OF THE SUB-BOXES AND THE NUMBER OF SUB-BOXES ON EACH LINE.

    """
    max_ll_pos = np.max(ll_pos)
    min_ll_pos = np.min(ll_pos)
    range_ll_pos = np.arange(min_ll_pos,max_ll_pos + segdist,segdist)
    base = list(product(range_ll_pos, repeat =2))
    line_list = {}
    len_line_list = []
    for i in range(len(base)):
        if direction == 'z':
            group = ll_pos[(ll_pos[:,0:2] == base[i]).all(axis=1)]
        elif direction == 'x':
            group = ll_pos[(ll_pos[:,1:] == base[i]).all(axis=1)]
        elif direction == 'y':
            group = ll_pos[(ll_pos[:,(0,2)] == base[i]).all(axis=1)]
        if len(group) > 1:
            group_diff = np.round(np.diff(group,axis=0),decimals=10)
            if direction == 'z':
                group_bool = (group_diff == np.array([0,0,segdist])).all(axis=1)
            elif direction == 'x':
                group_bool = (group_diff == np.array([segdist,0,0])).all(axis=1)
            elif direction == 'y':
                group_bool = (group_diff == np.array([0,segdist,0])).all(axis=1)
            group_bool = np.append(True,group_bool)
            group_divider = np.where(~group_bool)[0]
            start_idx = 0
            group_sep = []
            for i in group_divider:
                group_sep.append(group[start_idx:i])
                start_idx = i
            group_sep.append(group[start_idx:len(group)])
            line_list[base[i]] = group_sep
            #if len(group_sep) > 1:
            #    for j in range(len(group_sep)):
            #        line_list[base[i]] = group_sep[j][0]
            #else:
            #    line_list[base[i]] = group_sep[0]
            #line_list += group_sep
        else:
            #line_list += [group]
            line_list[base[i]] = [group]
    for i in range(len(line_list.values())):
        if len(list(line_list.values())[i]) <= 1:
            len_line_list.append(list(line_list.values())[i][0].shape[0])
        else:
            len_line_sub = []
            for n in range(len(list(line_list.values())[i])):
                len_line_sub.append(list(line_list.values())[i][n].shape[0])
            len_line_list.append(len_line_sub)
    return line_list, len_line_list
    
def find_slab(start_line, line_list, segdist, direction = 'x'):
    """
    Parameters
    ----------
    start_line : list of coordinates
        THE LIST OF SUB-BOXES THAT FORM A 1D LINE. WE WANT TO EXPAND THIS INTO A 2D SLAB
        
    direction: the initial expansion that we start our search
        'x' = START WITH 1D ON X, EXPAND TO 2D ON X AND Y, THEN EXPAND TO 3D
        'y' = START WITH 1D ON Y, EXPAND TO 2D ON Y AND Z, THEN EXPAND TO 3D
        'z' = START WITH 1D ON Z, EXPAND TO 2D ON Z AND X, THEN EXPAND TO 3D

    Returns
    -------
    slab : lowerleft and upper right coordinate of a slab
        THE SLAB AFTER WE EXPAND THE 1D LINE TO ACHIEVE MAXIMUM AREA.

    """
    #Format of a slab, a list containing the lower-left coordinate and the upper-right coordinate
    #Initially, the slab is a line. Then we expand it.
    slab = [list(start_line[0]),list(start_line[-1]+segdist)]
    
    if direction == 'x':
        line_x_list = line_list
    elif direction == 'y':
        line_y_list = line_list
    elif direction == 'z':
        line_z_list = line_list
    
    stop_top = 0
    stop_bottom = 0
    stop_all = 0
    
    #START THE LOOP
    while stop_all == 0:
        slab_volume = volume_cal(slab[0],slab[1])
        #start_line_base = list(line_x_list.keys())[np.argmax(len_line_x_list)]
        #start_line_base = (slab[0][0],slab[0][1])
        
        #-------------------------------------------------------------------------------------------------------------
        #SCENARIO 1: START WITH 1D ON X, EXPAND TO 2D ON X AND Y
        if direction == 'x':
            top_line_base = (slab[1][1], slab[1][2] - segdist) #use the upper-right block to calculate (increase y value - first element in the tuple)
                
            #bottom_line_base = (start_line_base[0]-segdist,start_line_base[1])
            if top_line_base in list(line_x_list.keys()) and line_x_list[top_line_base][0].shape[0] > 0:   
                top_line = line_x_list[top_line_base][0]
                slab_top_ll_x = max(slab[0][0],np.min(top_line[:,0]))
                slab_top_ur_x = min(slab[1][0] - segdist,np.max(top_line[:,0])) + segdist
                slab_top = [[slab_top_ll_x,slab[0][1],slab[0][2]],[slab_top_ur_x,top_line_base[0]+segdist,top_line_base[1]+segdist]]
                slab_top_volume = volume_cal(slab_top[0],slab_top[1])
            else:
                stop_top = 1 #if the expansion reaches the top limit, stop the expasion in this direction
                slab_top_volume = slab_volume
            
            bottom_line_base = (slab[0][1] - segdist, slab[0][2]) #use the lower-left block to calculate (decrease y value - first element in the tuple)
            if bottom_line_base in list(line_x_list.keys()) and line_x_list[bottom_line_base][0].shape[0] > 0: 
                bottom_line = line_x_list[bottom_line_base][0]
                slab_bottom_ll_x = max(slab[0][0],np.min(bottom_line[:,0]))
                slab_bottom_ur_x = min(slab[1][0] - segdist,np.max(bottom_line[:,0])) + segdist
                slab_bottom = [[slab_bottom_ll_x,bottom_line_base[0],bottom_line_base[1]],[slab_bottom_ur_x,slab[1][1],slab[1][2]]]
                slab_bottom_volume = volume_cal(slab_bottom[0],slab_bottom[1])
            else:
                stop_bottom = 1 #if the expansion reaches the bottom limit, stop the expasion in this direction
                slab_bottom_volume = slab_volume
                
        #-------------------------------------------------------------------------------------------------------------
        #SCENARIO 2: START WITH 1D ON Y, EXPAND TO 2D ON Y AND Z
        if direction == 'y':
            top_line_base = (slab[1][0] - segdist, slab[1][2]) #use the upper-right block to calculate  (increase z value - second element in the tuple)
                
            if top_line_base in list(line_y_list.keys()) and line_y_list[top_line_base][0].shape[0] > 0:   
                top_line = line_y_list[top_line_base][0]
                slab_top_ll_y = max(slab[0][1],np.min(top_line[:,1]))
                slab_top_ur_y = min(slab[1][1] - segdist,np.max(top_line[:,1])) + segdist
                slab_top = [[slab[0][0],slab_top_ll_y,slab[0][2]],[top_line_base[0]+segdist,slab_top_ur_y,top_line_base[1]+segdist]]
                slab_top_volume = volume_cal(slab_top[0],slab_top[1])
            else:
                stop_top = 1 #if the expansion reaches the top limit, stop the expasion in this direction
                slab_top_volume = slab_volume
            
            bottom_line_base = (slab[0][0], slab[0][2] - segdist)  #use the bottom-left block to calculate  (decrease z value - second element in the tuple)
            if bottom_line_base in list(line_y_list.keys()) and line_y_list[bottom_line_base][0].shape[0] > 0: 
                bottom_line = line_y_list[bottom_line_base][0]
                slab_bottom_ll_y = max(slab[0][1],np.min(bottom_line[:,1]))
                slab_bottom_ur_y = min(slab[1][1] - segdist,np.max(bottom_line[:,1])) + segdist
                slab_bottom = [[bottom_line_base[0],slab_bottom_ll_y,bottom_line_base[1]],[slab[1][0],slab_bottom_ur_y,slab[1][2]]]
                slab_bottom_volume = volume_cal(slab_bottom[0],slab_bottom[1])
            else:
                stop_bottom = 1 #if the expansion reaches the bottom limit, stop the expasion in this direction
                slab_bottom_volume = slab_volume                
        #-------------------------------------------------------------------------------------------------------------
        #SCENARIO 3: START WITH 1D ON Z, EXPAND TO 2D ON Z AND X
        if direction == 'z':
            top_line_base = (slab[1][0], slab[1][1] - segdist) #use the upper-right block to calculate  (increase x value - first element in the tuple; keep y value)
                
            if top_line_base in list(line_z_list.keys()) and line_z_list[top_line_base][0].shape[0] > 0:   
                top_line = line_z_list[top_line_base][0]
                slab_top_ll_z = max(slab[0][2],np.min(top_line[:,2]))
                slab_top_ur_z = min(slab[1][2] - segdist,np.max(top_line[:,2])) + segdist
                slab_top = [[slab[0][0],slab[0][1],slab_top_ll_z],[top_line_base[0]+segdist,top_line_base[1]+segdist,slab_top_ur_z]]
                slab_top_volume = volume_cal(slab_top[0],slab_top[1])
            else:
                stop_top = 1 #if the expansion reaches the top limit, stop the expasion in this direction
                slab_top_volume = slab_volume
            
            bottom_line_base = (slab[0][0] - segdist, slab[0][1])  #use the bottom-left block to calculate  (decrease x value - first element in the tuple; keep y value)
            if bottom_line_base in list(line_z_list.keys()) and line_z_list[bottom_line_base][0].shape[0] > 0: 
                bottom_line = line_z_list[bottom_line_base][0]
                slab_bottom_ll_z = max(slab[0][2],np.min(bottom_line[:,2]))
                slab_bottom_ur_z = min(slab[1][2] - segdist,np.max(bottom_line[:,2])) + segdist
                slab_bottom = [[bottom_line_base[0],bottom_line_base[1],slab_bottom_ll_z],[slab[1][0],slab[1][1],slab_bottom_ur_z]]
                slab_bottom_volume = volume_cal(slab_bottom[0],slab_bottom[1])
            else:
                stop_bottom = 1 #if the expansion reaches the bottom limit, stop the expasion in this direction
                slab_bottom_volume = slab_volume                
        #-------------------------------------------------------------------------------------------------------------           
        
        if stop_bottom == 1 and stop_top == 1: #if in the beginning, there is no parallel line for the line to expand to 2D -> stop the loop
            stop_all = 1

        if slab_top_volume >= slab_bottom_volume and slab_top_volume > slab_volume:
            slab = slab_top
        elif slab_bottom_volume >= slab_top_volume and slab_bottom_volume > slab_volume:
            slab = slab_bottom
        else:
            stop_all = 1             
      
    return slab, volume_cal(slab[0], slab[1])

def find_box(start_slab, start_slab_thirdaxis, slab_list, segdist, direction = 'x'):
    """

    Parameters
    ----------
    start_slab : list of 2 coordinates
        The lower-left and the upper right coordinate of the slab that we start expanding.
    start_slab_thirdaxis : int
        The third-dimension axis value of the slab. (direction = 'x' -> thirdaxis is z, direction = 'y' -> thirdaxis is x, direction = 'z' -> thirdaxis is y)
    slab_list:
        The list of slabs to evaluate
    direction: the initial expansion that we start our search
        'x' = START WITH 1D ON X, EXPAND TO 2D ON X AND Y, THEN EXPAND TO 3D
        'y' = START WITH 1D ON Y, EXPAND TO 2D ON Y AND Z, THEN EXPAND TO 3D
        'z' = START WITH 1D ON Z, EXPAND TO 2D ON Z AND X, THEN EXPAND TO 3D

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    #the initial box is the biggest slab
    box = start_slab
    
    stop_top = 0
    stop_bottom = 0
    stop_all = 0
    
    while stop_all == 0:
        box_volume = volume_cal(box[0],box[1])
        
        #-------------------------------------------------------------------------------------------------------------
        #SCENARIO 1: START WITH 1D ON X, EXPAND TO 2D ON X AND Y 
        if direction == 'x':
            top_slab_z = start_slab_thirdaxis+segdist
            if top_slab_z in list(slab_list.keys()):
                top_slab = slab_list[top_slab_z]
                #stack the top slab to the current slab to create a box
                box_top_ll_x = max(box[0][0],top_slab[0][0])
                box_top_ur_x = min(box[1][0],top_slab[1][0])
                box_top_ll_y = max(box[0][1],top_slab[0][1])
                box_top_ur_y = min(box[1][1],top_slab[1][1])
                box_top = [[box_top_ll_x,box_top_ll_y,start_slab_thirdaxis],[box_top_ur_x,box_top_ur_y,top_slab_z+segdist]]
                box_top_volume = volume_cal(box_top[0],box_top[1])
            else:
                stop_top = 1
                box_top_volume = box_volume
            
            bottom_slab_z = start_slab_thirdaxis-segdist
            if bottom_slab_z in list(slab_list.keys()):
                bottom_slab = slab_list[bottom_slab_z]
                #stack the bottom slab to the current slab to create a box
                box_bottom_ll_x = max(box[0][0],bottom_slab[0][0])
                box_bottom_ur_x = min(box[1][0],bottom_slab[1][0])
                box_bottom_ll_y = max(box[0][1],bottom_slab[0][1])
                box_bottom_ur_y = min(box[1][1],bottom_slab[1][1])
                box_bottom = [[box_bottom_ll_x,box_bottom_ll_y,bottom_slab_z],[box_bottom_ur_x,box_bottom_ur_y,start_slab_thirdaxis+segdist]]
                box_bottom_volume = volume_cal(box_bottom[0],box_bottom[1])
            else:
                stop_bottom = 1
                box_bottom_volume = box_volume

        #-------------------------------------------------------------------------------------------------------------
        #SCENARIO 2: START WITH 1D ON Y, EXPAND TO 2D ON Y AND Z                
        if direction == 'y':
            top_slab_x = start_slab_thirdaxis+segdist
            if top_slab_x in list(slab_list.keys()):
                top_slab = slab_list[top_slab_x]
                #stack the top slab to the current slab to create a box
                box_top_ll_y = max(box[0][1],top_slab[0][1])
                box_top_ur_y = min(box[1][1],top_slab[1][1])
                box_top_ll_z = max(box[0][2],top_slab[0][2])
                box_top_ur_z = min(box[1][2],top_slab[1][2])
                box_top = [[start_slab_thirdaxis,box_top_ll_y,box_top_ll_z],[top_slab_x+segdist,box_top_ur_y,box_top_ur_z]]
                box_top_volume = volume_cal(box_top[0],box_top[1])
            else:
                stop_top = 1
                box_top_volume = box_volume
            
            bottom_slab_x = start_slab_thirdaxis-segdist
            if bottom_slab_x in list(slab_list.keys()):
                bottom_slab = slab_list[bottom_slab_x]
                #stack the bottom slab to the current slab to create a box
                box_bottom_ll_y = max(box[0][1],bottom_slab[0][1])
                box_bottom_ur_y = min(box[1][1],bottom_slab[1][1])
                box_bottom_ll_z = max(box[0][2],bottom_slab[0][2])
                box_bottom_ur_z = min(box[1][2],bottom_slab[1][2])
                box_bottom = [[bottom_slab_x,box_bottom_ll_y,box_bottom_ll_z],[start_slab_thirdaxis+segdist,box_bottom_ur_y,box_bottom_ur_z]]
                box_bottom_volume = volume_cal(box_bottom[0],box_bottom[1])
            else:
                stop_bottom = 1
                box_bottom_volume = box_volume

        #-------------------------------------------------------------------------------------------------------------
        #SCENARIO 3: START WITH 1D ON Z, EXPAND TO 2D ON Z AND X                
        if direction == 'z':
            top_slab_y = start_slab_thirdaxis+segdist
            if top_slab_y in list(slab_list.keys()):
                top_slab = slab_list[top_slab_y]
                #stack the top slab to the current slab to create a box
                box_top_ll_z = max(box[0][2],top_slab[0][2])
                box_top_ur_z = min(box[1][2],top_slab[1][2])
                box_top_ll_x = max(box[0][0],top_slab[0][0])
                box_top_ur_x = min(box[1][0],top_slab[1][0])
                box_top = [[box_top_ll_x,start_slab_thirdaxis,box_top_ll_z],[box_top_ur_x,top_slab_y+segdist,box_top_ur_z]]
                box_top_volume = volume_cal(box_top[0],box_top[1])
            else:
                stop_top = 1 #the expansion hits the upper limit
                box_top_volume = box_volume
            
            bottom_slab_y = start_slab_thirdaxis-segdist
            if bottom_slab_y in list(slab_list.keys()):
                bottom_slab = slab_list[bottom_slab_y]
                #stack the bottom slab to the current slab to create a box
                box_bottom_ll_z = max(box[0][2],bottom_slab[0][2])
                box_bottom_ur_z = min(box[1][2],bottom_slab[1][2])
                box_bottom_ll_x = max(box[0][0],bottom_slab[0][0])
                box_bottom_ur_x = min(box[1][0],bottom_slab[1][0])
                box_bottom = [[box_bottom_ll_x,bottom_slab_y,box_bottom_ll_z],[box_bottom_ur_x,start_slab_thirdaxis+segdist,box_bottom_ur_z]]
                box_bottom_volume = volume_cal(box_bottom[0],box_bottom[1])
            else:
                stop_bottom = 1 #the expansion hits lower upper limit
                box_bottom_volume = box_volume

        #-------------------------------------------------------------------------------------------------------------
        
        if stop_top == 1 and stop_bottom == 1: #if in the beginning, there is no parallel slab for the slab to expand to 3D -> stop the loop
            stop_all = 1
        
        if box_top_volume >= box_bottom_volume and box_top_volume > box_volume:
            box = box_top
        elif box_bottom_volume >= box_top_volume and box_bottom_volume > box_volume:
            box = box_bottom
        else:
            stop_all = 1
        """
        if stop_top != 1: 
            if box_top_volume >= box_bottom_volume and box_top_volume > box_volume:
                box_new = box_top
        if stop_bottom != 1:
            if box_bottom_volume >= box_top_volume and box_bottom_volume > box_volume:
                box_new = box_bottom
        if stop_top == 1 and stop_bottom == 1: #if in the beginning, there is no parallel slab for the slab to expand to 3D -> stop the loop
            stop_all = 1
        else: #this means the box can still expand on at least 1 directtion
            if box == box_new: #this mean the box cannot expand to increase the volume -> stop the loop
                stop_all = 1
            else:
                box = box_new #if the box can expand to increase the volume, update the box
        """
        return box, volume_cal(box[0], box[1])
    
def find_maximized_region(pos,segdist):
    ll_pos = pos[:,0,:]
    
    line_x_list, len_line_x_list = find_lines(ll_pos, segdist, direction='x')
    line_y_list, len_line_y_list = find_lines(ll_pos, segdist, direction='y')
    line_z_list, len_line_z_list = find_lines(ll_pos, segdist, direction='z')
    
    max_ll_pos = np.max(ll_pos)
    min_ll_pos = np.min(ll_pos)
    range_ll_pos = np.arange(min_ll_pos,max_ll_pos + segdist,segdist)
    
    #---------------------------------------------------------------------------------------------------------------
    #SCENARIO 1: START WITH 1D ON X, EXPAND TO 2D ON X AND Y, THEN EXPAND TO 3D
    #The goal here is to find the slab on each z-axis first, then we stack those slabs together to find the biggest volume
    start_line_each_slab = {}
    for z in range_ll_pos: #loop through the z axis
        temp = line_x_list[(min_ll_pos,z)][0] #temp is a starting start_line so we can compare which line is longest on the x-y plane
        for y in range_ll_pos: #loop through the y axis
            for k in range(len(line_x_list[(y,z)])): #loop throug the number of lines in each x-y plane
                if line_x_list[(y,z)][k].shape[0] >= temp.shape[0]:
                    temp = line_x_list[(y,z)][k]
        if temp.shape[0] > 0: #avoid region with no line to start with           
            start_line_each_slab[z] = temp #we don't take into account the non-continuous slab here -> it is already taken into account when we evaluate the other two dimensions
    
    #start_line = list(line_x_list.values())[np.argmax(len_line_x_list)][0]
    slab_list = {}
    slab_vol_list = []
    for z in list(start_line_each_slab.keys()):
        start_line = start_line_each_slab[z]
        slab_list[z] = find_slab(start_line, line_x_list, segdist, direction = 'x')[0]
        slab_vol_list.append(find_slab(start_line, line_x_list, segdist, direction = 'x')[1])
        
    start_slab = list(slab_list.values())[np.argmax(slab_vol_list)]
    start_slab_z = list(slab_list.keys())[np.argmax(slab_vol_list)]
    
    box_x, box_vol_x = find_box(start_slab, start_slab_z, slab_list, segdist, direction = 'x')                
    
    #---------------------------------------------------------------------------------------------------------------
    #SCENARIO 2: START WITH 1D ON Y, EXPAND TO 2D ON Y AND Z, THEN EXPAND TO 3D
    start_line_each_slab = {}
    for x in range_ll_pos: #loop through the x axis
        temp = line_y_list[(x,min_ll_pos)][0] #temp is a starting start_line so we can compare which line is longest on the y-z plane
        for z in range_ll_pos: #loop through the z axis
            for k in range(len(line_y_list[(x,z)])): #loop throug the number of lines in each x-y plane
                if line_y_list[(x,z)][k].shape[0] >= temp.shape[0]:
                    temp = line_y_list[(x,z)][k]
        if temp.shape[0] > 0: #avoid region with no line to start with
            start_line_each_slab[x] = temp #we don't take into account the non-continuous slab here -> it is already taken into account when we evaluate the other two dimensions
    
    slab_list = {}
    slab_vol_list = []
    for x in list(start_line_each_slab.keys()):
        start_line = start_line_each_slab[x]
        slab_list[x] = find_slab(start_line, line_y_list, segdist, direction = 'y')[0]
        slab_vol_list.append(find_slab(start_line, line_y_list, segdist, direction = 'y')[1])
    
    start_slab = list(slab_list.values())[np.argmax(slab_vol_list)]
    start_slab_x = list(slab_list.keys())[np.argmax(slab_vol_list)]
    
    box_y, box_vol_y = find_box(start_slab, start_slab_x, slab_list, segdist, direction = 'y')  
    
    #---------------------------------------------------------------------------------------------------------------
    #SCENARIO 3: START WITH 1D ON Z, EXPAND TO 2D ON X AND Z, THEN EXPAND TO 3D
    start_line_each_slab = {}
    for y in range_ll_pos: #loop through the y axis
        temp = line_z_list[(min_ll_pos,y)][0] #temp is a starting start_line so we can compare which line is longest on the y-z plane
        for x in range_ll_pos: #loop through the x axis
            for k in range(len(line_z_list[(x,y)])): #loop throug the number of lines in each x-y plane
                if line_z_list[(x,y)][k].shape[0] >= temp.shape[0]:
                    temp = line_z_list[(x,y)][k]
        if temp.shape[0] > 0: #avoid region with no line to start with
            start_line_each_slab[y] = temp #we don't take into account the non-continuous slab here -> it is already taken into account when we evaluate the other two dimensions
    
    slab_list = {}
    slab_vol_list = []
    for y in list(start_line_each_slab.keys()):
        start_line = start_line_each_slab[y]
        slab_list[y] = find_slab(start_line, line_z_list, segdist, direction = 'z')[0]
        slab_vol_list.append(find_slab(start_line, line_z_list, segdist, direction = 'z')[1])
    
    start_slab = list(slab_list.values())[np.argmax(slab_vol_list)]
    start_slab_y = list(slab_list.keys())[np.argmax(slab_vol_list)]
    
    box_z, box_vol_z = find_box(start_slab, start_slab_y, slab_list, segdist, direction = 'z')  
    
    #----------------------------------------------------------------------------------------------------------------
    box_list = [box_x, box_y, box_z]
    box_vol_list = [box_vol_x, box_vol_y, box_vol_z]
    box_max = box_list[np.argmax(box_vol_list)]
    return box_max    

def extend_initial_refined_region(init_refined_region, segdist, mlim_list, lim_index, code_name, ds):
    refined_region = np.ones((2,3)) #create a placeholder for the refined region
    #locating the boxes surrounding the initial refined region
    surrounding_boxes = []
    direction_list = ['pos_x','neg_x','pos_y','neg_y','pos_z','neg_z']
    for direction in direction_list: #build the surrounding boxes in 6 directions (to allow parallelization)
        if direction == 'pos_x':
            surround_ll = np.array([init_refined_region[1][0], init_refined_region[0][1] - segdist, init_refined_region[0][2] - segdist])
            surround_ur = np.array([init_refined_region[1][0] + segdist, init_refined_region[1][1] + segdist, init_refined_region[1][2] + segdist])
            surrounding_boxes.append([direction,surround_ll,surround_ur])
        elif direction == 'neg_x':
            surround_ll = np.array([init_refined_region[0][0] - segdist, init_refined_region[0][1] - segdist, init_refined_region[0][2] - segdist])
            surround_ur = np.array([init_refined_region[0][0], init_refined_region[1][1] + segdist, init_refined_region[1][2] + segdist])
            surrounding_boxes.append([direction,surround_ll,surround_ur])
        elif direction == 'pos_y':
            surround_ll = np.array([init_refined_region[0][0] - segdist, init_refined_region[1][1], init_refined_region[0][2] - segdist])
            surround_ur = np.array([init_refined_region[1][0] + segdist, init_refined_region[1][1] + segdist, init_refined_region[1][2] + segdist])
            surrounding_boxes.append([direction,surround_ll,surround_ur])
        elif direction == 'neg_y':
            surround_ll = np.array([init_refined_region[0][0] - segdist, init_refined_region[0][1] - segdist, init_refined_region[0][2] - segdist])
            surround_ur = np.array([init_refined_region[1][0] + segdist, init_refined_region[0][1], init_refined_region[1][2] + segdist])
            surrounding_boxes.append([direction,surround_ll,surround_ur])
        elif direction == 'pos_z':
            surround_ll = np.array([init_refined_region[0][0] - segdist, init_refined_region[0][1] - segdist, init_refined_region[1][2]])
            surround_ur = np.array([init_refined_region[1][0] + segdist, init_refined_region[1][1] + segdist, init_refined_region[1][2] + segdist])
            surrounding_boxes.append([direction,surround_ll,surround_ur])
        elif direction == 'neg_z':
            surround_ll = np.array([init_refined_region[0][0] - segdist, init_refined_region[0][1] - segdist, init_refined_region[0][2] - segdist])
            surround_ur = np.array([init_refined_region[1][0] + segdist, init_refined_region[1][1] + segdist, init_refined_region[0][2]])
            surrounding_boxes.append([direction,surround_ll,surround_ur])

    my_storage = {}
    for sto, i in yt.parallel_objects(range(len(surrounding_boxes)), nprocs, storage = my_storage):
        direction = surrounding_boxes[i][0]
        ll = surrounding_boxes[i][1]
        ur = surrounding_boxes[i][2]
        #buffer =(segdist/100)*ds.units.code_length #set buffer when loading the box
        #reg = ds.box(ll-buffer, ur+buffer)
        reg = ds.box(ll, ur)

        if code_name == 'GEAR':
            dm = ParticleUnion("DarkMatter",["PartType5","PartType2"])
            ds.add_particle_union(dm)
        if code_name == 'GADGET3':
            dm = ParticleUnion("DarkMatter",["PartType5","PartType1"])
            ds.add_particle_union(dm)
        if code_name == 'AREPO' or code_name == 'GIZMO':
            dm = ParticleUnion("DarkMatter",["PartType2","PartType1"])
            ds.add_particle_union(dm)

        lr_name_dict = {'GEAR': 'DarkMatter', 'GADGET3': 'DarkMatter', 'AREPO': 'DarkMatter', 'GIZMO': 'DarkMatter', 'RAMSES': 'DM', 'ART': 'darkmatter', 'CHANGA': 'DarkMatter'}
        
        dm_m = reg[(lr_name_dict[code_name],'particle_mass')].to('Msun').v
        dm_pos = reg[(lr_name_dict[code_name],'particle_position')].to('code_length').v

        lr_dm_pos = dm_pos[dm_m > np.sort(list(set(mlim_list)))[lim_index]] 
        
        if direction == 'pos_x':
            refined_region[1][0] = np.min(lr_dm_pos[:,0])
        elif direction == 'neg_x':
            refined_region[0][0] = np.max(lr_dm_pos[:,0])
        elif direction == 'pos_y':
            refined_region[1][1] = np.min(lr_dm_pos[:,1])
        elif direction == 'neg_y':
            refined_region[0][1] = np.max(lr_dm_pos[:,1])
        elif direction == 'pos_z':
            refined_region[1][2] = np.min(lr_dm_pos[:,2])
        elif direction == 'neg_z':
            refined_region[0][2] = np.max(lr_dm_pos[:,2])
    return refined_region

#-----------------------------------------------------------------------------------------
#Main code
#code_name = sys.argv[1]
#directory = sys.argv[2]
code_name = 'GADGET3'
directory = '/scratch/bbvl/tnguyen2/sandbox/refined_region_optz/'
lim_index = 1 #refined region up to the second highest level (0 for first, 1 for second, 2 for third, etc.)

ds = yt.load(directory + 'snapshot_010/snapshot_010.0.hdf5')
seg_pos_list, mlim_list, mtype_list, segdist = reduce_range(code_name, directory, ds, ds.domain_left_edge.v, ds.domain_right_edge.v)
refined_bool = np.logical_and(mlim_list == np.sort(list(set(mlim_list)))[lim_index], mtype_list <= lim_index + 1) #assuming that there is no subbox with uncontinuous levels of refinement (for example, a subbox with level 1 and 3; without level 2)

while sum(refined_bool) == 0: #if there is no refined region found, reduce the range until we find one
    reduced_bool = np.logical_and(mlim_list == np.sort(list(set(mlim_list)))[lim_index], mtype_list <= lim_index + 2)
    reduced_pos = seg_pos_list[reduced_bool]
    reduced_pos = np.round(reduced_pos,decimals=10)
    reduced_region = find_maximized_region(reduced_pos,segdist)
    seg_pos_list, mlim_list, mtype_list, segdist = reduce_range(code_name, directory, ds, reduced_region[0], reduced_region[1])
    refined_bool = np.logical_and(mlim_list == np.sort(list(set(mlim_list)))[lim_index], mtype_list <= lim_index + 1)

refined_pos = seg_pos_list[refined_bool]
refined_pos = np.round(refined_pos,decimals=10) #just to make sure there is no floating point error
init_refined_region = find_maximized_region(refined_pos,segdist)
refined_region = extend_initial_refined_region(init_refined_region, segdist, mlim_list, lim_index, code_name, ds)

if yt.is_root():
    np.save(directory + 'refined_region.npy', refined_region) #save the refined region

