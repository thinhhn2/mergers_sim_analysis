#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 22:05:47 2024

@author: thinhnguyen
"""
import numpy as np
import matplotlib.pyplot as plt
import ytree

#STEPS TO TRACE THE MISSING BRANCH IN AREPO
#1. Make plot showing the top 4 biggest halos around the main halo at a snapshot 
#where the two merging halos are still separated. Double check with the star particle plot
#2. Make ROCKSTAR plots until the secondary halo disappear. Check its halo ID before it disappear
#3. Go to consistent-tree, use command 
#halo_secondary_189 = list(arbor.select_halos("(tree['forest', 'halo_id'] == 670)  & (tree['forest', 'Snap_idx'] == 189)"))
#to identify the secondary halo in the consistent-tree output
#4. Use ytree to trace whether the secondary halo merge with the first halo (meaning its 
#descendent belong to the main halo progenitor line). Make consistent-tree plot if necessary.

#Make ROCKSTAR halo plots 
data = np.load('halotree_Thinh_structure_refinedboundary_2nd.npy',allow_pickle=True).tolist()

index = '193'
halo = data['0'][index]
time = halo['time']
coor = np.array(halo['coor'])
rvir = halo['Rvir']

coor_min = coor - 1.5*rvir
coor_max = coor + 1.5*rvir

halolist = np.genfromtxt('out_%s.list' % index)
xlist = halolist[:,8]
ylist = halolist[:,9]
zlist = halolist[:,10]
idlist = halolist[:,0]
masslist = halolist[:,2]
rvirlist = halolist[:,5]/1000
poslist = np.array([xlist,ylist,zlist]).T

boolean = (poslist < coor_max).all(axis=1) & (poslist > coor_min).all(axis=1)
pos_selected = poslist[boolean]
rvir_selected = rvirlist[boolean]
mass_selected = masslist[boolean]
id_selected = idlist[boolean]

pos_plot = pos_selected[np.argsort(-rvir_selected)][:5]
rvir_plot = rvir_selected[np.argsort(-rvir_selected)][:5]
mass_plot = mass_selected[np.argsort(-rvir_selected)][:5]
id_plot = id_selected[np.argsort(-rvir_selected)][:5]

color = plt.cm.plasma(np.linspace(0, 1, len(rvir_plot)))

fig, ax = plt.subplots(figsize=(9,9))
ax.set_xlim((coor_min[0],coor_max[0]))
ax.set_ylim((coor_min[1],coor_max[1]))
for i in range(len(pos_plot)):
    circle = plt.Circle((pos_plot[i][0],pos_plot[i][1]), rvir_plot[i], color=color[i],alpha=0.5)
    ax.add_patch(circle)
    ax.text(pos_plot[i][0],pos_plot[i][1],'%d' % id_plot[i])

ax.set_xlabel('Particle Position X (code length)', fontsize=14)
ax.set_ylabel('Particle Position Y (code length)', fontsize=14)
ax.tick_params(axis='both', labelsize=14)
ax.set_title('Current time: %.2f Gyr' % time, fontsize=14)
plt.savefig('rockstar_halos_%s.png' % index, dpi = 400)
#---------------------------------------------------------------------------
#Code to track the halo in the consistent-tree outputs
#For example, in this case, at timestep 180, the primary halo has ID = 390, the secondary halo has ID = 442
arbor = ytree.load('arbor/arbor.h5')
halo_primary_180 = list(arbor.select_halos("(tree['forest', 'halo_id'] == 390)  & (tree['forest', 'Snap_idx'] == 180)"))
halo_second_180 = list(arbor.select_halos("(tree['forest', 'halo_id'] == 442)  & (tree['forest', 'Snap_idx'] == 180)"))

halo_primary = halo_primary_180[0]
halo_second = halo_second_180[0]

time_counter = 180

#This is for skipping the timestep without making the plots
for i in range(16):
    halo_primary = halo_primary.descendent
    halo_second = halo_second.descendent
    print(np.log10(halo_second['virial_radius'].to('unitary').v))
    print(np.log10(halo_second['mass'].to('Msun').v))
    time_counter += 1

for i in range(3): #make plot till timestep 193 (193 - 180 + 1 = 14)
    fig, ax = plt.subplots(figsize=(9,9))
    x_primary = halo_primary['position_x'].v*60
    y_primary = halo_primary['position_y'].v*60
    rvir_primary = halo_primary['virial_radius'].to('unitary').v*60
    id_primary = halo_primary['halo_id']
    #--------------------------------------------------------------------------------
    x_second = halo_second['position_x'].v*60
    y_second = halo_second['position_y'].v*60
    rvir_second = halo_second['virial_radius'].to('unitary').v*60
    id_second = halo_second['halo_id']
    #---------------------------------------------------------------------------------
    ax.set_xlim((x_primary - 1.5*rvir_primary, x_primary + 1.5*rvir_primary))
    ax.set_ylim((y_primary - 1.5*rvir_primary, y_primary + 1.5*rvir_primary))
    circle_primary = plt.Circle((x_primary,y_primary), rvir_primary, color='grey',alpha=0.5)
    circle_second = plt.Circle((x_second,y_second), rvir_second, color='red',alpha=0.5)
    ax.add_patch(circle_primary)
    ax.add_patch(circle_second)
    ax.text(x_primary, y_primary, '%d' % id_primary)
    ax.text(x_second, y_second, '%d' % id_second)
    ax.set_xlabel('Particle Position X (code length)', fontsize=14)
    ax.set_ylabel('Particle Position Y (code length)', fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    plt.savefig('consistenttree_halos_%s.png' % time_counter, dpi = 400)
    #---------------------------------------------------------------------------------
    halo_primary = halo_primary.descendent
    halo_second = halo_second.descendent
    time_counter += 1
    

    