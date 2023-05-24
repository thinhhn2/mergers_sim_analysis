#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 14:06:36 2023

@author: thinhnguyen
"""

"""
Kirk's halo list is a dictory type that has each key represent the halos in each tree/branch.
This code convert it into a dictory that has each key present the halos in each redshift
This is to make the parallelism run faster (each processor now only deals with 1 snapshot)
"""

import numpy as np

#Load the Kirk's reduced halo list
halo_s =np.load('halolist_shield.npy',allow_pickle=True).tolist()
halo_ns =np.load('halolist_nonshield.npy',allow_pickle=True).tolist()

#Remove the keys that are not the halo lists
del_list = ['pos','prog_found','rootlist','rootMvir','rvir','uids']

for i in del_list:
    del halo_s[i]
    del halo_ns[i]

#This represents the redshift. The index corresponds to the index in the pfs.dat file    
z_index = np.arange(0,132,1)    

#Create the new dictionaries    
halo_by_z_s = {}
halo_by_z_ns = {}

#Find all the halos with a specific redshift and add them into the new dictories with 
#the corresponding keys/redshifts. We do this step for the shield and the nonshield

#Shield simulation
for i in z_index:
    halos_each = []
    for mainkey, value in halo_s.items():
        if i in value.keys():
            value_add = value[i]
            value_add.append(mainkey)
            halos_each.append(value_add)
    key_name = str(i)
    halo_by_z_s[key_name] = halos_each

#Nonshield simulation
for i in z_index:
    halos_each = []
    for mainkey, value in halo_ns.items():
        if i in value.keys():
            value_add = value[i]
            value_add.append(mainkey)
            halos_each.append(value_add)
    key_name = str(i)
    halo_by_z_ns[key_name] = halos_each

#Save the new files
np.save('halolist_by_redshift_shield.npy',halo_by_z_s)
np.save('halolist_by_redshift_nonshield.npy',halo_by_z_ns)
        
    
    
    