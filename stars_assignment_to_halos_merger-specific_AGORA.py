import numpy as np
import yt
from astropy.constants import G
import os
import glob as glob
from scipy.spatial.distance import cdist
import time as time_sys
import collections
import sys
import healpy as hp
from merger_compute import merger_compute

def convert_unitary_to_codelength(tree, factor):
    if factor == 'CHANGA':
        tree_c = {}
        for branch in tree.keys():
            tree_c[branch] = {}
            for idx in tree[branch].keys():
                tree_c[branch][idx] = {}
                tree_c[branch][idx]['uid'] = tree[branch][idx]['uid']
                tree_c[branch][idx]['Halo_Center'] = tree[branch][idx]['Halo_Center'] - 0.5
                tree_c[branch][idx]['Vel_Com'] = tree[branch][idx]['Vel_Com']
                tree_c[branch][idx]['Halo_Radius'] = tree[branch][idx]['Halo_Radius']
                tree_c[branch][idx]['Halo_Mass'] = tree[branch][idx]['Halo_Mass']
                tree_c[branch][idx]['time'] = tree[branch][idx]['time']
    else:
        tree_c = {}
        for branch in tree.keys():
            tree_c[branch] = {}
            for idx in tree[branch].keys():
                tree_c[branch][idx] = {}
                tree_c[branch][idx]['uid'] = tree[branch][idx]['uid']
                tree_c[branch][idx]['Halo_Center'] = tree[branch][idx]['Halo_Center']*factor
                tree_c[branch][idx]['Vel_Com'] = tree[branch][idx]['Vel_Com']*factor
                tree_c[branch][idx]['Halo_Radius'] = tree[branch][idx]['Halo_Radius']*factor
                tree_c[branch][idx]['Halo_Mass'] = tree[branch][idx]['Halo_Mass']
                tree_c[branch][idx]['time'] = tree[branch][idx]['time']
    return tree_c

def search_closest_upper(value, array):
    diff = array - value
    return np.where(diff >= 0)[0][0]

def extract_and_order_snapshotIdx(rawtree, branch):
    #this function extract only the snapshot key (i.e. the integer value) from the rawtree halotree output
    keys = list(rawtree[branch].keys())
    snapshotIdx = [x for x in keys if not isinstance(x, str)]
    snapshotIdx.sort()
    return snapshotIdx

def extract_position_radius_mass_vel(rawtree, branch):
    idx_list = extract_and_order_snapshotIdx(rawtree, branch)
    radius_list = np.array([])
    mass_list = np.array([])
    position_list = np.empty(shape=(0,3))  
    vel_list = np.empty(shape=(0,3))
    for idx in idx_list:
        radius_list = np.append(radius_list, rawtree[branch][idx]['Halo_Radius'])
        mass_list = np.append(mass_list, rawtree[branch][idx]['Halo_Mass'])
        position_list = np.vstack((position_list, rawtree[branch][idx]['Halo_Center']))
        vel_list = np.vstack((vel_list, rawtree[branch][idx]['Vel_Com']))
    return position_list, radius_list, mass_list, vel_list, idx_list

def list_of_halos_wstars_idx(rawtree_s, pos_allstars, idx):
    halo_wstars_pos = np.empty(shape=(0,3))
    halo_wstars_rvir = np.array([])
    halo_wstars_branch = np.array([])
    for branch, vals in rawtree_s.items():
        if idx in vals.keys():
            if (np.linalg.norm(pos_allstars - rawtree_s[branch][idx]['Halo_Center'], axis=1) < rawtree_s[branch][idx]['Halo_Radius']).any():
                halo_wstars_pos = np.vstack((halo_wstars_pos, vals[idx]['Halo_Center']))
                halo_wstars_rvir = np.append(halo_wstars_rvir, vals[idx]['Halo_Radius'])
                halo_wstars_branch = np.append(halo_wstars_branch, branch)   
    return halo_wstars_pos, halo_wstars_rvir, halo_wstars_branch

def vecs_calc(nside):
    pix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside,np.arange(pix))
    vecs = hp.ang2vec(np.array(theta),np.array(phi))
    return vecs

def cut_particles(pos,mass,center,cut_size=700,dense=False,segments=1,timing=0):
    if timing:
        time5 = time_sys.time()
    bool = np.full(len(pos),True)
    vec = {}
    if not dense:
        vec[0] = vecs_calc(1)
        vec[1] = vecs_calc(1)
        vec[2] = vecs_calc(1)
    else:
        vec[0] = vecs_calc(1)
        vec[1] = vecs_calc(2)
        vec[2] = vecs_calc(2)
    dist_pos = np.linalg.norm(pos-center,axis=1)
    inner = np.array([40,20,12])
    annuli = np.linspace(0,dist_pos.max()/3,inner[0])
    annuli2 = np.linspace(dist_pos.max()/3,2*dist_pos.max()/3,inner[1])[1:]
    annuli3 = np.linspace(2*dist_pos.max()/3,dist_pos.max(),inner[2])[1:]
    annuli = np.append(annuli,annuli2)
    annuli = np.append(annuli,annuli3)
    index = np.arange(len(pos))
    for i in range(len(annuli)-1):
        bool_in_0 = (dist_pos <= annuli[i+1])*(dist_pos > annuli[i])
        cutlength = cut_size*annuli[i+1]/annuli[-1]
        current_group = np.arange(len(inner))[(i >= np.cumsum(inner)-inner)*(i < np.cumsum(inner))][0]
        pos_norm = dist_pos[bool_in_0][:,np.newaxis]
        pos_in = pos[bool_in_0]
        vec_ang = np.dot((pos_in-center),vec[current_group].T)
        if len(np.unique(vec_ang)) != len(vec_ang):
            vec_ang += vec_ang.min()*1e-5*np.random.random(len(vec_ang[0]))[np.newaxis,:]
        #pos_group = np.searchsorted(vec_ang,vec_ang.max(axis=1))
        pos_group = np.where(vec_ang == vec_ang.max(axis=1)[:,np.newaxis])[1]
        # lenpos = len(pos_group)
        # lenind = len(index[bool_in_0])
        # if lenpos > lenind:
        #     print('Mismatch',lenpos,lenind,vec_ang.shape,pos_in.shape,len(bool_in_0))
        #     print(np.where(vec_ang == vec_ang.max(axis=1)[:,np.newaxis])[1].shape)
        #     pos_group = pos_group[:-(lenpos-lenind)]
        for t in range(len(vec[current_group])):
            current_index = index[bool_in_0][pos_group==t]
            if len(current_index) > max(cutlength,max(10/segments,1)):
                mass_tot = mass[current_index].sum()
                cut = int(np.ceil(len(current_index)/cutlength))
                bool[current_index] = False
                rand_ind = np.random.choice(current_index,size=len(current_index),replace=False)
                bool[rand_ind[0::cut]] = True
                mass_in = mass[rand_ind[0::cut]].sum()
                mass[rand_ind[0::cut]] *= mass_tot/mass_in
        if timing and time_sys.time()-time5 >timing:
            print('Make Annuli',time_sys.time()-time5)
            time5 = time_sys.time()
    del pos
    mass[index[np.logical_not(bool)]] *= 1e-10
    return mass[bool], bool

def find_total_E(star_pos, star_vel, ds, rawtree_s, branch, idx, dense=True):
    #
    #This function finds the total energy of an array of star particles in one halo at a certain timestep.
    if star_pos.shape == (3,): #reshaping the star_pos and star_vel to be 2D arrays, in the case of a single star
        star_pos = star_pos.reshape(1,3)
        star_vel = star_vel.reshape(1,3)
    #
    if os.path.exists(metadata_dir + '/cutparticles') == False:
        os.mkdir(metadata_dir + '/cutparticles')
    if os.path.exists(metadata_dir + '/cutparticles/cutparticles_Branch_%s_idx_%s.npy' % (branch,idx)):
        cutparticles = np.load(metadata_dir + '/cutparticles/cutparticles_Branch_%s_idx_%s.npy' % (branch,idx), allow_pickle=True).tolist()
        posA_cut = cutparticles['pos']
        massA_cut = cutparticles['mass']
    else:
        regA = ds.sphere(rawtree_s[branch][idx]['Halo_Center'], rawtree_s[branch][idx]['Halo_Radius'])
        massA = regA['all','particle_mass'].to('kg').v
        posA = regA['all','particle_position'].to('m')
        #velA = regA['all','particle_velocity'].to('m/s')
        #
        boolloc = np.linalg.norm(posA.to('code_length').v - rawtree_s[branch][idx]['Halo_Center'], axis=1) <= rawtree_s[branch][idx]['Halo_Radius']
        #
        posA = posA[boolloc]
        #velA = velA[boolall]
        massA = massA[boolloc]
        del boolloc
        #
        if len(massA) > 10000:
            #
            if dense == True:
                density = 2
            else:
                density = 1
            #
            lenmass = len(massA)
            segments = max(int(lenmass/5e6),1)
            cutmin = int(max(density*250*np.log10(len(massA))/6,density*100))
            cut_size = 700
            cut_size=min(max(cutmin/segments,1),max(cut_size/segments,1))
            #
            centerA = (rawtree_s[branch][idx]['Halo_Center']*ds.units.code_length).to('m').v
            massA_cut, boolA_cut = cut_particles(posA.v,massA,centerA, cut_size=cut_size, segments=segments)
            posA_cut = posA[boolA_cut]
            np.save(metadata_dir + '/cutparticles/cutparticles_Branch_%s_idx_%s.npy' % (branch,idx), {'pos':posA_cut, 'mass':massA_cut})
        else:
            massA_cut = massA
            posA_cut = posA
            np.save(metadata_dir + '/cutparticles/cutparticles_Branch_%s_idx_%s.npy' % (branch,idx), {'pos':posA_cut, 'mass':massA_cut})
        del posA, massA
    #
    """
    #use cdist, 100x faster
    disAinv = 1/cdist((star_pos*ds.units.code_length).to('m').v, posA.v, 'euclidean')
    disAinv[~np.isfinite(disAinv)] = 0
    disAinv[np.isnan(disAinv)] = 0
    #
    PE = np.sum(-G.value*massA.to('kg').v*disAinv, axis=1)
    velcom = (rawtree_s[branch][idx]['Vel_Com']*ds.units.code_length/ds.units.s).to('m/s').v
    KE = 0.5*np.linalg.norm(star_vel - velcom, axis=1)**2
    E = KE + PE
    E[np.isnan(E)] = 1e99
    """
    #use cdist, 100x faster
    if len(posA_cut) == 0:
        chunk_size = np.inf
    else:
        chunk_size = int((10*1e9/8)/len(posA_cut)) #this will take as much 10 GB, should use 30 GB when running
    if chunk_size >= len(star_pos):
        disAinv_cut = 1/cdist((star_pos*ds.units.code_length).to('m').v, posA_cut.v, 'euclidean')
        disAinv_cut[~np.isfinite(disAinv_cut)] = 0
        disAinv_cut[np.isnan(disAinv_cut)] = 0
        PE = np.sum(-G.value*massA_cut*disAinv_cut, axis=1)
    else: #Cut the star_pos array by chunks to save memory
        PE = np.array([])
        for j in range(0, len(star_pos), chunk_size):
            pos_chunk = star_pos[j : j + chunk_size, :]
            disAinv_cut = 1/cdist((pos_chunk*ds.units.code_length).to('m').v, posA_cut.v, 'euclidean')
            disAinv_cut[~np.isfinite(disAinv_cut)] = 0
            disAinv_cut[np.isnan(disAinv_cut)] = 0
            PE = np.append(PE, np.sum(-G.value*massA_cut*disAinv_cut, axis=1))
    #
    velcom = (rawtree_s[branch][idx]['Vel_Com']*ds.units.code_length/ds.units.s).to('m/s').v
    KE = 0.5*np.linalg.norm(star_vel - velcom, axis=1)**2
    E = KE + PE
    E[np.isnan(E)] = 1e99
    return E


def stars_assignment(rawtree_s, pfs, metadata_dir, codetp, print_mode = True):
    """
    This function uniquely assigns each star in the simulation box to a halo. 
    There are two steps:
    + Step 1: Locate the halo where a star is born in. If a star is born in the intersection of multiple halos, perform energy calculation to see which halo that star belongs to. Assume that that star remains in that halo until the end of the simulation. If that halo is a sub-halo, add that star to the main halo when the two halos merge. This step helps speed up the star assignment process because we don't need to calculate the orbital energy of each star.
    + Step 2: Re-evaluate the assumption and output from Step 1. If a star moves outside of the in-situ halo at a certain timestep (hereby called "reassign star"), remove that star from that halo, and find whether that star is bound to another halo. This steps require enegy calculation for each reassigned star, but the number of reassigned stars is much smaller than the total number of stars.
    ---
    Input
    ---
    rawtree_s: 
      the smoothed SHINBAD merger tree output
    pfs: 
      the list of the snapshot's directory
    halo_dir:
        the directory to the halo finding and the refined region output
    metadata_dir: 
      the directory to the file containing the star's metadata
    numsegs:
        the number of segments to divide the box into. This is used if the star metadata needs to be extracted
    print_mode:
        whether to print the output of the function (for debugging purpose)
    ---
    Output
    ---
    output_final: 
      a dictionary containing the halos with the ID of their stars. The keys of the dictionary are the Snapshot indices.
      Each snapshot index is another dictary whose keys are the branches with stars, the SFR, and the total stellar mass.
    """
    if glob.glob(metadata_dir + '/' + 'star_metadata_allbox_*.npy') == [] or os.path.exists(metadata_dir + '/' + 'stars_assignment_step1_backup_ProgBranch-%s.npy' % progenitor_branch) == False or os.path.exists(metadata_dir + '/' + 'halo_wstars_map_ProgBranch-%s.npy' % progenitor_branch) == False: 
        halo_wstars_map = {}
        output = {}
        for idx in range(0, len(pfs)):
            output[idx] = {}
        highvel_IDs = np.array([])
        starting_idx = 0
    else:
        halo_wstars_map = np.load(metadata_dir + '/' + 'halo_wstars_map_ProgBranch-%s.npy' % progenitor_branch, allow_pickle=True).tolist()
        output = np.load(metadata_dir + '/' + 'stars_assignment_step1_backup_ProgBranch-%s.npy' % progenitor_branch, allow_pickle=True).tolist()
        highvel_IDs = np.load(metadata_dir + '/' + 'highvel_IDs_ProgBranch-%s.npy' % progenitor_branch, allow_pickle=True).tolist()
        starting_idx = list(halo_wstars_map.keys())[-1] + 1
    #------------------------------------------------------------------------
    for idx in range(starting_idx, len(pfs)):
        #
        if (codetp == 'CHANGA' or codetp == 'ART') and os.path.exists(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % idx) == False:
            continue
        if codetp == 'AREPO' and idx == 5:
            continue
        metadata = np.load(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
        pos_all = metadata['pos']
        ID_all = metadata['ID'].astype(int)
        vel_all = metadata['vel']*1e3 #convert from km/s to m/s
        if idx == 0 or len(list(output[idx].keys())) == 0:
            ID_all_prev = np.array([])
        else:
            ID_all_prev = np.concatenate(list(output[idx].values())).astype(int) #these are the ID of the stars that are already assigned to halos in the previous snapshot. This also helps address the issue of a main progenitor branch ending before the last snapshot.
        #
        ID_unassign = np.setdiff1d(ID_all, ID_all_prev)
        pos_unassign = pos_all[np.intersect1d(ID_all, ID_unassign, return_indices=True)[1]]
        vel_unassign = vel_all[np.intersect1d(ID_all, ID_unassign, return_indices=True)[1]]
        del vel_all
        #Obtain the halos with stars
        halo_wstars_pos, halo_wstars_rvir, halo_wstars_branch = list_of_halos_wstars_idx(rawtree_s, pos_all, idx)
        del pos_all
        halo_wstars_map[idx] = {} #stored it for later used in Step 2 of the code
        halo_wstars_map[idx]['pos'] = halo_wstars_pos
        halo_wstars_map[idx]['rvir'] = halo_wstars_rvir
        halo_wstars_map[idx]['branch_wstars'] = halo_wstars_branch
        #
        #The shape of halo_boolean is (X,Y), where X is the number of star particles and Y is the number of halos with stars
        halo_boolean = np.linalg.norm(pos_unassign[:, np.newaxis, :] - halo_wstars_pos, axis=2) <= halo_wstars_rvir
        #The number of halos a star particle is in. For example, if this value = 2, the star particle is in the region of 2 halos
        overlap_boolean = np.sum(halo_boolean, axis=1) 
        #
        ID_overlap = ID_unassign[overlap_boolean > 1]
        ID_overlap = np.append(ID_overlap, np.intersect1d(ID_unassign,highvel_IDs)) #need to re-evaluate the highvel_IDs even if they are in the region of only 1 halo.
        ID_overlap = np.unique(ID_overlap)
        halo_boolean_overlap = halo_boolean[np.intersect1d(ID_unassign, ID_overlap, return_indices=True)[1]]
        #
        ID_indp = ID_unassign[overlap_boolean == 1]
        ID_indp = np.setdiff1d(ID_indp, highvel_IDs) #need to re-evaluate the highvel_IDs even if they are in the region of only 1 halo.
        halo_boolean_indp = halo_boolean[np.intersect1d(ID_unassign, ID_indp, return_indices=True)[1]]
        #
        #The list of stars in each halo's region
        starmap_ID = []
        for j in range(halo_boolean_indp.shape[1]):
            starmap_ID.append(ID_indp[halo_boolean_indp[:,j]])
        #
        if len(ID_overlap) > 0:
            if codetp == 'AREPO' or codetp == 'GADGET3':
                ds = yt.load(pfs[idx], unit_base = {"length": (1.0, "Mpccm/h")})
            else:
                ds = yt.load(pfs[idx])
            pos_overlap = pos_unassign[np.intersect1d(ID_unassign, ID_overlap, return_indices=True)[1]]
            vel_overlap = vel_unassign[np.intersect1d(ID_unassign, ID_overlap, return_indices=True)[1]]
            #overlap_energy_map is a dictionary that contains the energy of a star in each of its overlap regions
            overlap_energy_map = collections.defaultdict(list)
            #this for loop calculate the energy for all overlapped stars that are in the same halo to speed up time
            for i_branch in range(len(halo_wstars_branch)):
                #Select the IDs that are in the same halo and the same time step for the energy calculation
                ID_for_erg = ID_overlap[halo_boolean_overlap[:,i_branch]]
                if len(ID_for_erg) > 0:
                    pos_for_erg = pos_overlap[halo_boolean_overlap[:,i_branch]]
                    vel_for_erg = vel_overlap[halo_boolean_overlap[:,i_branch]]
                    E = find_total_E(pos_for_erg, vel_for_erg, ds, rawtree_s, halo_wstars_branch[i_branch], idx)
                    for k in range(len(ID_for_erg)):
                        overlap_energy_map[ID_for_erg[k]].append(E[k])
            for k in range(len(ID_overlap)):
                overlap_branch = halo_wstars_branch[halo_boolean_overlap[k]]
                E_list = overlap_energy_map[ID_overlap[k]]
                if len(E_list) == 0:
                    continue
                if np.min(E_list) < 0:
                    bound_branch = overlap_branch[np.argmin(E_list)]
                    starmap_ID[list(halo_wstars_branch).index(bound_branch)] = np.append(starmap_ID[list(halo_wstars_branch).index(bound_branch)], ID_overlap[k]) 
                    #print('For Star %s, the overlapped branches are %s and the energies are %s. This star is assigned to Branch %s.' % (int(ID_overlap[k]), overlap_branch, E_list, bound_branch))
                else:
                    #print('For Star %s, the overlapped branches are %s and the energies are %s. This star is NOT bound to any branches.' % (int(ID_overlap[k]), overlap_branch, E_list))
                    highvel_IDs = np.append(highvel_IDs, ID_overlap[k])
                    highvel_IDs = np.unique(highvel_IDs)
        len_starmap = [len(i) for i in starmap_ID]
        # Add stars to subsequent snapshots
        for i in range(len(halo_wstars_branch)):
            if len(starmap_ID[i]) > 0: 
                for j in extract_and_order_snapshotIdx(rawtree_s, halo_wstars_branch[i]): #assuming when a star forms inside a halo, it will not leave that halo 
                    if int(j) >= idx:
                        if halo_wstars_branch[i] not in output[j].keys():
                            output[j][halo_wstars_branch[i]] = starmap_ID[i]
                        else:
                            output[j][halo_wstars_branch[i]] = np.append(output[j][halo_wstars_branch[i]], starmap_ID[i])
                #for subbranch (or deeper sub-branch), the stars in that sub-branch will belong to the branch at lower level after the two halos merge
                nlevels = halo_wstars_branch[i].count('_')
                if nlevels > 1:
                    print('DEEPER SUB-BRANCHES DETECTED')
                loop_branch = halo_wstars_branch[i]
                for level in range(nlevels): #add the stars in the sub-branch to higher branches
                    deepest_lvl = loop_branch.split('_')[-1]
                    mainbranch = loop_branch.split('_' + deepest_lvl)[0]
                    merge_timestep = np.max(extract_and_order_snapshotIdx(rawtree_s, loop_branch)) + 1
                    last_timestep = np.max(extract_and_order_snapshotIdx(rawtree_s, mainbranch))
                    for j in range(merge_timestep, last_timestep + 1):
                        if mainbranch not in output[j].keys():
                            output[j][mainbranch] = starmap_ID[i]
                        else:
                            output[j][mainbranch] = np.append(output[j][mainbranch], starmap_ID[i])
                    loop_branch = mainbranch
        #
        np.save('%s/stars_assignment_step1_backup_ProgBranch-%s.npy' % (metadata_dir, progenitor_branch), output)
        np.save('%s/halo_wstars_map_ProgBranch-%s.npy' % (metadata_dir, progenitor_branch), halo_wstars_map)
        np.save('%s/highvel_IDs_ProgBranch-%s.npy' % (metadata_dir, progenitor_branch), highvel_IDs)
        #
        if print_mode == True:
            #print(idx, 'Number of total unassigned stars is:', len(ID_unassign))
            #print('Number of overlapped stars is', len(ID_overlap), ', Number of independent stars is', len(ID_indp))
            #print('Halo with stars:', halo_wstars_branch)
            print(idx, 'Number of assingned stars in each halo:', dict(zip(np.array(halo_wstars_branch)[np.array(len_starmap) != 0], np.array(len_starmap)[np.array(len_starmap) != 0])), '\n') #only print the halos with stars in them
        #Free some memory
        del metadata, ID_all, ID_unassign, pos_unassign, vel_unassign, halo_boolean, overlap_boolean, ID_indp, starmap_ID
        if len(ID_overlap) > 0:
            del ID_overlap, ds, pos_overlap, vel_overlap, overlap_energy_map
        try:
            del ID_for_erg, pos_for_erg, vel_for_erg, E, E_list
        except:
            None
    #------------------------------------------------------------------------
    #This step removes the stars that moves outside of the halo's virial radius and addes them to another halos if needed. 
    #The unique stellar mass and SFR is also calculated in this step. 
    #print(halo_wstars_map)
    #print(output)
    if os.path.exists(metadata_dir + '/' + 'stars_assignment_step2_backup_ProgBranch-%s.npy' % progenitor_branch) == False:
        output_final = {} #the re-analyzed output
        reassign_dict = {} #the list of stars that need to be re-assigned during step 2 of the code
        for idx in range(0, len(pfs)):
            reassign_dict[idx] = np.array([]).astype(int)
        prev_halo_map = collections.defaultdict(list) #the dictionary containing the branch each star belongs to in Step 1 of the code
        starting_idx_step2 = 0
    else:
        output_final = np.load(metadata_dir + '/' + 'stars_assignment_step2_backup_ProgBranch-%s.npy' % progenitor_branch, allow_pickle=True).tolist()
        reassign_dict = np.load(metadata_dir + '/' + 'reassign_dict_step2_ProgBranch-%s.npy' % progenitor_branch, allow_pickle=True).tolist()
        prev_halo_map = np.load(metadata_dir + '/' + 'prev_halo_map_step2_ProgBranch-%s.npy' % progenitor_branch, allow_pickle=True).tolist()
        starting_idx_step2 = list(output_final.keys())[-1] + 1
    for idx in range(starting_idx_step2, len(pfs)):
        output_final[idx] = {}
        if codetp == 'AREPO' or codetp == 'GADGET3':
            ds = yt.load(pfs[idx], unit_base = {"length": (1.0, "Mpccm/h")})
        else:
            ds = yt.load(pfs[idx])
        #
        if codetp == 'CHANGA' and os.path.exists(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % idx) == False:
            continue
        if codetp == 'AREPO' and idx == 5:
            continue
        metadata = np.load(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
        pos_all = metadata['pos']
        ID_all = metadata['ID'].astype(int)
        vel_all = metadata['vel']*1e3 #convert from km/s to m/s
        for branch in output[idx].keys():
            if idx not in extract_and_order_snapshotIdx(rawtree_s, branch):
                continue
            ID = output[idx][branch]
            #obtain the stars found in the initial output
            pos = pos_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            ID = ID_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            vel = vel_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            #try:
            #    del pos_all, vel_all
            #except:
            #    None
            #
            halo_center = rawtree_s[branch][idx]['Halo_Center']
            halo_radius = rawtree_s[branch][idx]['Halo_Radius']
            #
            #remain_bool: stars that still remain in the halo where they are born
            #loss_bool: stars that move out of the halo where they were born 
            remain_bool = np.linalg.norm(pos - halo_center, axis=1) < halo_radius
            loss_bool = np.linalg.norm(pos - halo_center, axis=1) >= halo_radius
            #------------------------
            #Reassign the "loss" stars to new halos by using bound energy condition. Note that when a star is lose, we will check its energy for the rest of the timestep. 
            ID_loss = ID[loss_bool]
            for ID_loss_i in ID_loss:
                if ID_loss_i not in prev_halo_map.keys():
                    prev_halo_map[ID_loss_i] = branch
            if len(ID_loss) > 0:
                for j in range(idx, max(extract_and_order_snapshotIdx(rawtree_s, branch)) + 1):
                    reassign_dict[j] = np.append(reassign_dict[j], ID_loss)
                    reassign_dict[j] = np.unique(reassign_dict[j])
            #------------------------
            ID_remain = ID[remain_bool]
            ID_remain = np.setdiff1d(ID_remain, reassign_dict[idx])
            output_final[idx][branch] = {}
            output_final[idx][branch]['ID'] = ID_remain
        #-------------------------
        #reassign_energy_map is a dictionary that contains the energy of a star gets outside of its first assigned halo and move to another halo region
        #The logic here is similar to how we calculate the energy for the overlapped stars
        if len(list(output[idx].keys())) == 0:
            pos_reassign = np.empty(shape=(0,3))
            vel_reassign = np.empty(shape=(0,3))
            ID_reassign = np.empty(0)
        else:
            reassign_energy_map = collections.defaultdict(list)
            pos_reassign = pos[np.intersect1d(ID, reassign_dict[idx], return_indices=True)[1]]
            vel_reassign = vel[np.intersect1d(ID, reassign_dict[idx], return_indices=True)[1]]
            ID_reassign = ID[np.intersect1d(ID, reassign_dict[idx], return_indices=True)[1]]
        print('At Snapshot', idx, ', %s stars need to be re-assigned' % len(reassign_dict[idx]))
        halo_wstars_pos, halo_wstars_rvir, halo_wstars_branch = halo_wstars_map[idx].values() #obtain the list of halos with stars, the halo_wstars_map is computed above
        halo_boolean_reassign = np.linalg.norm(pos_reassign[:, np.newaxis, :] - halo_wstars_pos, axis=2) <= halo_wstars_rvir
        for i_branch in range(len(halo_wstars_branch)):
            ID_for_erg = ID_reassign[halo_boolean_reassign[:,i_branch]]
            if len(ID_for_erg) > 0:
                pos_for_erg = pos_reassign[halo_boolean_reassign[:,i_branch]]
                vel_for_erg = vel_reassign[halo_boolean_reassign[:,i_branch]]
                E = find_total_E(pos_for_erg, vel_for_erg, ds, rawtree_s, halo_wstars_branch[i_branch], idx)
                for k in range(len(ID_for_erg)):
                    reassign_energy_map[ID_for_erg[k]].append(E[k])
        for k in range(len(ID_reassign)):
            reassign_branch = halo_wstars_branch[halo_boolean_reassign[k]] #these are the branches that the reassigned stars move to (before the reassignment and energy calculation)
            if ID_reassign[k] in reassign_energy_map.keys():
                E_list = reassign_energy_map[ID_reassign[k]]
                if np.min(E_list) < 0:
                    new_bound_branch = reassign_branch[np.argmin(E_list)]
                    #print('At Snapshot', idx, 'Star', ID_reassign[k], 'move from Branch', prev_halo_map[ID_reassign[k]], 'to', new_bound_branch)
                    if new_bound_branch not in output_final[idx].keys(): #add the stars bounded with the new halo to the output_final
                        output_final[idx][new_bound_branch] = {}
                        output_final[idx][new_bound_branch]['ID'] = np.array([ID_reassign[k]])
                    else:
                        output_final[idx][new_bound_branch]['ID'] = np.append(output_final[idx][new_bound_branch]['ID'], ID_reassign[k])
            else:
                continue #the star is not bound to any halo, skip this star  
        #Save for backup
        np.save('%s/stars_assignment_step2_backup_ProgBranch-%s.npy' % (metadata_dir, progenitor_branch), output_final)
        np.save('%s/reassign_dict_step2_ProgBranch-%s.npy' % (metadata_dir, progenitor_branch), reassign_dict)
        np.save('%s/prev_halo_map_step2_ProgBranch-%s.npy' % (metadata_dir, progenitor_branch), prev_halo_map)
    #Finalize the output_final star ID and calculate the unique total stellar mass and SFR.
    """
    for idx in output_final.keys():
        if codetp == 'CHANGA' and os.path.exists(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % idx) == False:
            continue
        if codetp == 'AREPO' and idx == 5:
            continue
        metadata = np.load(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
        mass_all = metadata['mass']
        age_all = metadata['age']
        ID_all = metadata['ID']
        pos_all = metadata['pos']
        vel_all = metadata['vel']
        met_all = metadata['met']
        for branch in output_final[idx].keys():
            ID = output_final[idx][branch]['ID']
            mass = mass_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            age = age_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            pos = pos_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            vel = vel_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            met = met_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            #
            output_final[idx][branch]['total_mass'] = np.sum(mass)
            output_final[idx][branch]['sfr'] = np.sum(mass[age < 0.01])/1e7
            output_final[idx][branch]['mass'] = mass
            output_final[idx][branch]['age'] = age
            output_final[idx][branch]['pos'] = pos
            output_final[idx][branch]['ID'] = ID
            output_final[idx][branch]['vel'] = vel
            output_final[idx][branch]['met'] = met
    """
    return output_final

def branch_first_rearrange(output_final):
    output_re = {}
    for snapshot in output_final.keys():
        for branch in output_final[snapshot].keys():
            if branch not in output_re.keys():
                output_re[branch] = {}
                output_re[branch][snapshot] = output_final[snapshot][branch]
            else:
                output_re[branch][snapshot] = output_final[snapshot][branch]
    return output_re

            
if __name__ == "__main__":
    halo_dir = sys.argv[1]
    metadata_dir = sys.argv[2]
    halotree_ver = sys.argv[3]
    progenitor_branch = sys.argv[4]
    codetp = metadata_dir.split('/work/hdd/bdax/tnguyen2/AGORA/')[1].split('/metadata')[0]
    #
    if codetp == 'GIZMO':
        rawtree = np.load(halo_dir + '/halotree_%s_final_corr.npy' % halotree_ver, allow_pickle=True).tolist()
    else:
        rawtree = np.load(halo_dir + '/halotree_%s_final.npy' % halotree_ver, allow_pickle=True).tolist()
    RCT = False
    #rawtree_s = np.load('/work/hdd/bdax/tnguyen2/AGORA/%s/Reformatting_consistenttree_output_checkpoints/halotrees_RCT_reformatted.npy' % codetp, allow_pickle=True).tolist()
    pfsfile = np.loadtxt(halo_dir + '/pfs_allsnaps_%s.txt' % halotree_ver, dtype=str)
    pfs = pfsfile[:,0]
    print('Done loading data')
    print(metadata_dir)
    #
    #merger_key = merger_compute(progenitor_branch, rawtree_s, 0.05, pfsfile)[4]
    #merger_key = np.append(progenitor_branch, merger_key)
    if codetp == 'ART':
        merger_key = ['0', '0_76', '0_25', '2', '0_16']
    elif codetp == 'ENZO':
        merger_key = ['0', '99', '0_63', '0_44', '4', '0_18']
    elif codetp == 'GADGET3':
        merger_key = ['0', '4', '0_89', '2', '1']
    elif codetp == 'AREPO':
        merger_key = ['0', '0_138', '0_148', '0_104', '2', '0_76']
    elif codetp == 'CHANGA':
        merger_key = ['0', '0_53', '0_40', '0_42']
    elif codetp == 'GIZMO':
        merger_key = ['0', '0_58_6', '0_58', '3', '2']
    elif codetp == 'GEAR':
        merger_key = ['0', '70', '0_93', '2', '3', '0_31']
    rawtree_merger = {}
    for key in merger_key:
        rawtree_merger[key] = rawtree[key]
    if RCT == True:
        if codetp == 'GIZMO' or codetp == 'GEAR':
            factor = 60000
        elif codetp == 'AREPO' or codetp == 'GADGET3':
            factor = 60
        elif codetp == 'ENZO' or codetp == 'RAMSES' or codetp == 'ART':
            factor = 1
        elif codetp == 'CHANGA':
            factor = 'CHANGA'
        rawtree_merger = convert_unitary_to_codelength(rawtree_merger, factor)
    #
    stars_assign_output = stars_assignment(rawtree_merger, pfs, metadata_dir, codetp, print_mode = True)
    np.save(metadata_dir + '/stars_assignment_ProgBranch-%s_snapFirst.npy' % progenitor_branch, stars_assign_output)
    #
    #This is to re-arange the data structure to match with Kirk's pipeline
    branch_first = True
    if branch_first == True:
        stars_assign_output_re = branch_first_rearrange(stars_assign_output)
        np.save(metadata_dir + '/stars_assignment_ProgBranch-%s_branchFirst.npy' % progenitor_branch, stars_assign_output_re)
