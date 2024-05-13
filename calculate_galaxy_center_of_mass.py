#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:04:10 2024

@author: thinhnguyen
"""
import numpy as np
import matplotlib.pyplot as plt

data = np.load('stars_123_branch-0.npy',allow_pickle=True).tolist()

center = np.array(data['com_coor_star'])
coor_all = np.array(data['coor'])
mass_all = np.array(data['mass'])

r_all = np.linalg.norm(coor_all - center,axis=1)
r_galaxy = 0.15*np.max(r_all) #the galaxy is in 0.15Rvir
r_lim = 0.15*np.max(r_all) 

counter = 0

while counter < 5:
    coor = coor_all[r_all < r_lim]
    mass = mass_all[r_all < r_lim]
    r_all = np.linalg.norm(coor_all - center,axis=1) #recalculate the distance with the new center of mass
    center_new = np.average(coor, weights=mass, axis=0)
    if np.linalg.norm(center_new - center) < 0.005:
        counter += 1
        center = center_new
    else:
        counter = 0
        center = center_new
    r_lim *= 0.975

#Plotting
fig, ax = plt.subplots(figsize=(9,9))
ax.scatter(coor_all[:,0], coor_all[:,1],s=0.2,alpha=0.5)
ax.scatter(center[0], center[1], s = 25, color='red')
circle = plt.Circle((center[0], center[1]), r_galaxy, color='red', fill=False)
ax.add_patch(circle)