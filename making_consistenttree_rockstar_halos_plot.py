#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 15:08:10 2024

@author: thinhnguyen
"""
import numpy as np
import matplotlib.pyplot as plt
import ytree

#data = np.load('halotree_Thinh_structure_refinedboundary_2nd.npy',allow_pickle=True).tolist()
data = np.load('/scratch/bbvl/tnguyen2/ENZO/halotree_Thinh_structure_with_com.npy',allow_pickle=True).tolist()

arbor = ytree.load('/scratch/bbvl/tnguyen2/ENZO/rockstar_halos/trees/arbor/arbor.h5')
#Identify the progenitor lineage of the main halo
prog_line = list(arbor[0]['prog'])

for halo in prog_line:
    time = halo['time']
    coor = np.array(halo['coor'])
    rvir = halo['Rvir']
    coor_min = coor - 5*rvir
    coor_max = coor + 5*rvir
    #
    halolist = np.genfromtxt('out_%s.list' % halo['Snap_idx'])
    xlist = halolist[:,8]
    ylist = halolist[:,9]
    zlist = halolist[:,10]
    idlist = halolist[:,0]
    masslist = halolist[:,2]
    rvirlist = halolist[:,5]/1000
    poslist = np.array([xlist,ylist,zlist]).T
    #Select the halos within 5Rvir of the main halos
    boolean = (poslist < coor_max).all(axis=1) & (poslist > coor_min).all(axis=1)
    pos_selected = poslist[boolean]
    rvir_selected = rvirlist[boolean]
    mass_selected = masslist[boolean]
    id_selected = idlist[boolean]
    #Select the top N massive halos within 5Rvir to plot
    N_plot = 20
    pos_plot = pos_selected[np.argsort(-rvir_selected)][:N_plot]
    rvir_plot = rvir_selected[np.argsort(-rvir_selected)][:N_plot]
    mass_plot = mass_selected[np.argsort(-rvir_selected)][:N_plot]
    id_plot = id_selected[np.argsort(-rvir_selected)][:N_plot]
    #
    color = plt.cm.plasma(np.linspace(0, 1, len(rvir_plot)))
    #
    fig, ax = plt.subplots(figsize=(9,9))
    ax.set_xlim((coor_min[0],coor_max[0]))
    ax.set_ylim((coor_min[1],coor_max[1]))
    for i in range(len(pos_plot)):
        circle = plt.Circle((pos_plot[i][0],pos_plot[i][1]), rvir_plot[i], color=color[i],alpha=0.5)
        ax.add_patch(circle)
        ax.text(pos_plot[i][0],pos_plot[i][1],'%d' % id_plot[i])
    #
    ax.set_xlabel('Particle Position X (code length)', fontsize=14)
    ax.set_ylabel('Particle Position Y (code length)', fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_title('Current time: %.2f Gyr' % time, fontsize=14)
    plt.savefig('/scratch/bbvl/tnguyen2/ENZO/consistenttree_rockstars_halo_plots/rockstar_halos_%s.png' % halo['Snap_idx'], dpi = 300)