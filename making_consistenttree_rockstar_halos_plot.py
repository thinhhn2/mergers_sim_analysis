#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 15:08:10 2024

@author: thinhnguyen
"""
import numpy as np
import matplotlib.pyplot as plt
import ytree
import os

codetp = 'CHANGA'

if codetp == 'ENZO':
    data = np.load('/scratch/bbvl/tnguyen2/ENZO/halotree_Thinh_structure_with_com.npy',allow_pickle=True).tolist()
else:
    data = np.load('/scratch/bbvl/tnguyen2/%s/halotree_Thinh_structure_refinedboundary_2nd.npy' % codetp,allow_pickle=True).tolist()

if not os.path.exists('/scratch/bbvl/tnguyen2/%s/consistenttree_rockstars_halo_plots' % codetp):
    os.mkdir('/scratch/bbvl/tnguyen2/%s/consistenttree_rockstars_halo_plots' % codetp)

arbor = ytree.load('/scratch/bbvl/tnguyen2/%s/rockstar_halos/trees/arbor/arbor.h5' % codetp)
#Identify the progenitor lineage of the main halo
prog_line = list(arbor[0]['prog'])
scalestxt = np.genfromtxt('/scratch/bbvl/tnguyen2/%s/rockstar_halos/outputs/scales.txt' % codetp)

for halo in prog_line:
    #if os.path.exists('/scratch/bbvl/tnguyen2/%s/consistenttree_rockstars_halo_plots/rockstar_halos_%s.png' % (codetp, halo['Snap_idx'])):
    #    continue
    time = halo['time'].to('Gyr').v.tolist()
    coor = halo['position'].v
    rvir = halo['virial_radius'].to('unitary').v.tolist()
    coor_min = coor - 5*rvir
    coor_max = coor + 5*rvir
    #
    rockstar_idx = int(scalestxt[halo['Snap_idx']][0])
    #ART'S ROCKSTAR INDEX MISMATCHES WITH CONSISTENTTREE'S OUTPUT STARTING FROM SNAPSHOT 122. 
    #CONSISTENTTREE INDEX 121 = ROCKSTAR INDEX 121
    #CONSISTENTTREEE INDEX 122 = ROCKSTAR INDEX 123
    #CONSISTENTTREEE INDEX 123 = ROCKSTAR INDEX 124, etc.
    if codetp == 'ART_v18' and rockstar_idx >= 122:
        halolist = np.genfromtxt('/scratch/bbvl/tnguyen2/%s/rockstar_halos/out_%s.list' % (codetp, rockstar_idx+1))
    else:
        halolist = np.genfromtxt('/scratch/bbvl/tnguyen2/%s/rockstar_halos/out_%s.list' % (codetp, rockstar_idx))
    xlist = halolist[:,8]/60
    ylist = halolist[:,9]/60
    zlist = halolist[:,10]/60
    idlist = halolist[:,0]
    masslist = halolist[:,2]
    rvirlist = (halolist[:,5]/1000)/60
    poslist = np.array([xlist,ylist,zlist]).T
    #Select the halos within 5Rvir of the main halos
    boolean = (poslist < coor_max).all(axis=1) & (poslist > coor_min).all(axis=1)
    pos_selected = poslist[boolean]
    rvir_selected = rvirlist[boolean]
    mass_selected = masslist[boolean]
    id_selected = idlist[boolean]
    #Select the top N massive halos within 5Rvir to plot
    N_plot = 20
    if len(id_selected) > 20:
        pos_plot = pos_selected[np.argsort(-mass_selected)][:N_plot]
        rvir_plot = rvir_selected[np.argsort(-mass_selected)][:N_plot]
        mass_plot = mass_selected[np.argsort(-mass_selected)][:N_plot]
        id_plot = id_selected[np.argsort(-mass_selected)][:N_plot]
    else:
        pos_plot = pos_selected[np.argsort(-mass_selected)]
        rvir_plot = rvir_selected[np.argsort(-mass_selected)]
        mass_plot = mass_selected[np.argsort(-mass_selected)]
        id_plot = id_selected[np.argsort(-mass_selected)]
    #
    color = plt.cm.plasma(np.linspace(0, 1, len(rvir_plot)))
    #
    fig, ax = plt.subplots(figsize=(9,9))
    ax.set_xlim((coor_min[0],coor_max[0]))
    ax.set_ylim((coor_min[1],coor_max[1]))
    for i in range(len(pos_plot)):
        circle = plt.Circle((pos_plot[i][0],pos_plot[i][1]), rvir_plot[i], color=color[i],alpha=0.5)
        ax.add_patch(circle)
        if i == 0 or i == 1:
            ax.text(pos_plot[i][0],pos_plot[i][1],'%d' % id_plot[i], weight = 'bold', fontsize=14)
        else:
            ax.text(pos_plot[i][0],pos_plot[i][1],'%d' % id_plot[i])
    #
    ax.set_xlabel('Particle Position X (code length)', fontsize=14)
    ax.set_ylabel('Particle Position Y (code length)', fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_title('Current time: %.2f Gyr' % time, fontsize=14)
    plt.savefig('/scratch/bbvl/tnguyen2/%s/consistenttree_rockstars_halo_plots/rockstar_halos_%s.png' % (codetp, halo['Snap_idx']), dpi = 300)
    
#----------------------------------------------------------------------------------------------------
#Trace back the lineage of the secondary galaxies before merging with the primary galaxy
halo_primary = arbor[0]
merger_order = 1
if codetp == 'ENZO':
    if merger_order == 1:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 287)  & (tree['forest', 'Snap_idx'] == 141)"))[0]
    if merger_order == 2:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 470)  & (tree['forest', 'Snap_idx'] == 193)"))[0]
    if merger_order == 3:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 441)  & (tree['forest', 'Snap_idx'] == 208)"))[0]
    if merger_order == 4:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 1133)  & (tree['forest', 'Snap_idx'] == 249)"))[0]

if codetp == 'AREPO':
    if merger_order == 1:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 351)  & (tree['forest', 'Snap_idx'] == 133)"))[0]
    if merger_order == 2:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 496)  & (tree['forest', 'Snap_idx'] == 195)"))[0]
    if merger_order == 3:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 655)  & (tree['forest', 'Snap_idx'] == 203)"))[0]
    if merger_order == 4:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 1858)  & (tree['forest', 'Snap_idx'] == 271)"))[0]

if codetp == 'GADGET3':
    if merger_order == 1:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 362)  & (tree['forest', 'Snap_idx'] == 133)"))[0]
    if merger_order == 2:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 343)  & (tree['forest', 'Snap_idx'] == 204)"))[0]
    if merger_order == 3:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 304)  & (tree['forest', 'Snap_idx'] == 199)"))[0]
    if merger_order == 4:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 2041)  & (tree['forest', 'Snap_idx'] == 285)"))[0]

if codetp == 'GEAR':
    if merger_order == 1:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 228)  & (tree['forest', 'Snap_idx'] == 394)"))[0]
    if merger_order == 2:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 454)  & (tree['forest', 'Snap_idx'] == 535)"))[0]
    if merger_order == 3:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 672)  & (tree['forest', 'Snap_idx'] == 581)"))[0]
    if merger_order == 4:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 1918)  & (tree['forest', 'Snap_idx'] == 792)"))[0]

if codetp == 'GIZMO':
    if merger_order == 1:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 260)  & (tree['forest', 'Snap_idx'] == 140)"))[0]
    if merger_order == 2:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 694)  & (tree['forest', 'Snap_idx'] == 201)"))[0]
    if merger_order == 3:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 836)  & (tree['forest', 'Snap_idx'] == 211)"))[0]

if codetp == 'CHANGA':
    if merger_order == 1:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 337)  & (tree['forest', 'Snap_idx'] == 108)"))[0]
    if merger_order == 2:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 551)  & (tree['forest', 'Snap_idx'] == 190)"))[0]
    if merger_order == 3:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 327)  & (tree['forest', 'Snap_idx'] == 217)"))[0]

if codetp == 'ART_v18':
    if merger_order == 1:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 262)  & (tree['forest', 'Snap_idx'] == 159)"))[0]
    if merger_order == 2:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 469)  & (tree['forest', 'Snap_idx'] == 220)"))[0]
    if merger_order == 3:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 500)  & (tree['forest', 'Snap_idx'] == 241)"))[0]

if codetp == 'RAMSES':
    if merger_order == 1:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 1951)  & (tree['forest', 'Snap_idx'] == 137)"))[0]
    if merger_order == 2:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 4062)  & (tree['forest', 'Snap_idx'] == 201)"))[0]
    if merger_order == 3:    
        halo_second = list(arbor.select_halos("(tree['forest', 'halo_id'] == 1791)  & (tree['forest', 'Snap_idx'] == 206)"))[0]
