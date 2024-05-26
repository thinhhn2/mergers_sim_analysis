#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:04:10 2024

@author: thinhnguyen
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import yt
import glob as glob

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

branch = '0'

tree_data = np.load('halotree_Thinh_structure_with_com.npy',allow_pickle=True).tolist()
if branch == '0':
    snapshot_list = tree_data[branch].keys()
else:
    snapshot_list = []
    file_list = glob.glob('metadata/branch-%s/stars_*' % branch)
    for file in file_list:
        temp = file.split('stars_')[1]
        snapshot_list.append(temp.split('.npy')[0])

my_storage= {}

for sto, snapshot in yt.parallel_objects(snapshot_list, nprocs-1, storage = my_storage):
    pfs = np.loadtxt('pfs_manual.dat',dtype=str)
    ds = yt.load(pfs[int(snapshot)])
    
    star_data = np.load('metadata/branch-%s/stars_%s.npy' % (branch, snapshot),allow_pickle=True).tolist()
    #star_data = np.load('stars_239_branch-0.npy',allow_pickle=True).tolist()
    rvir = star_data[branch][snapshot]['Rvir']*ds.units.code_length
    rvir = rvir.to('kpc').v
    
    center = np.array(star_data['com_coor_star'])
    coor_all = np.array(star_data['coor'])
    mass_all = np.array(star_data['mass'])
    ID_all = np.array(star_data['ID']).astype(int)
    ft_all = np.array(star_data['formation_time'])
    vel_all = np.array(star_data['vel'])
    r_all = np.linalg.norm(coor_all - center,axis=1)
    
    r_15 = 0.15*rvir #the galaxy is in 0.15Rvir
    r_lim = r_15
    
    counter = 0
    Nstar = len(mass_all)
    
    while counter < 5 and Nstar > 500:
        coor = coor_all[r_all < r_lim]
        mass = mass_all[r_all < r_lim]
        Nstar = len(mass)
        
        center_new = np.average(coor, weights=mass, axis=0)
        
        if np.linalg.norm(center_new - center) < 0.005:
            counter += 1
            center = center_new
        else:
            counter = 0
            center = center_new
        r_all = np.linalg.norm(coor_all - center,axis=1) #recalculate the distance with the new center of mass
        r_lim *= 0.975    
    
    #Plotting
    fig, ax = plt.subplots(figsize=(9,9))
    ax.scatter(coor_all[:,0], coor_all[:,1],s=0.2,alpha=0.5)
    ax.scatter(center[0], center[1], s = 25, color='red')
    circle = plt.Circle((center[0], center[1]), r_15, color='red', fill=False)
    ax.add_patch(circle)
    
    r_all = np.linalg.norm(coor_all - center,axis=1)
    coor_galaxy = coor_all[r_all < r_15]
    mass_galaxy = mass_all[r_all < r_15]
    ID_galaxy = ID_all[r_all < r_15]
    ft_galaxy = ft_all[r_all < r_15]
    r_galaxy = r_all[r_all < r_15]
    vel_galaxy = vel_all[r_all < r_15]
    
    #currentime = data[branch][snapshot]['time']
    #sf_timescale = 0.01 #Gyr
    #sfr_15 = mass_galaxy[ft_galaxy > currenttime - sf_timescale].sum()/(sf_timescale*1e9)
    output15 = {}
    output15['center'] = center
    output15['sm15'] = mass_galaxy
    output15['coor15'] = coor_galaxy
    output15['vel15'] = vel_galaxy
    output15['id15'] = ID_galaxy
    output15['formation_time15'] = ft_galaxy
    
    np.save('metadata/branch-%s/stars15Rvir_%s.npy' % (branch,snapshot), output15)
