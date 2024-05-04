#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 21:35:07 2024

@author: thinhnguyen
"""

import yt
import numpy as np
from yt.data_objects.unions import ParticleUnion
from yt.data_objects.particle_filters import add_particle_filter

def extract_dm(index, codetp, halotree_file):
    tree = np.load(halotree_file,allow_pickle=True).tolist()
    pfs = np.loadtxt('pfs_manual.dat',dtype = str)
    if codetp == 'GADGET3' or codetp == 'AREPO':
        ds = yt.load(pfs[index], unit_base = {"length": (1.0, "Mpccm/h")})
    else:
        ds = yt.load(pfs[index])
        
    if codetp == 'ENZO':
        def darkmatter_init(pfilter, data):
            filter_darkmatter0 = np.logical_or(data["all", "particle_type"] == 1, data["all", "particle_type"] == 4)
            filter_darkmatter = np.logical_and(filter_darkmatter0,data['all', 'particle_mass'].to('Msun') > 1)
            return filter_darkmatter
        add_particle_filter("DarkMatter",function=darkmatter_init,filtered_type='all',requires=["particle_type","particle_mass"])
        ds.add_particle_filter("DarkMatter")

    #combine less-refined particles and refined-particles into one field for GEAR, GIZMO, AREPO, and GADGET3
    if codetp == 'GEAR':
        dm = ParticleUnion("DarkMatter",["PartType5","PartType2"])
        ds.add_particle_union(dm)
    if codetp == 'GADGET3':
        dm = ParticleUnion("DarkMatter",["PartType5","PartType1"])
        ds.add_particle_union(dm)
    if codetp == 'AREPO' or codetp == 'GIZMO':
        dm = ParticleUnion("DarkMatter",["PartType2","PartType1"])
        ds.add_particle_union(dm)
        
    coor = tree['0'][str(index)]['coor']
    rvir = tree['0'][str(index)]['Rvir']
    reg = ds.sphere(coor,(rvir,'code_length'))
    
    dm_name_dict = {'ENZO':'DarkMatter','GEAR': 'DarkMatter', 'GADGET3': 'DarkMatter', 'AREPO': 'DarkMatter', 'GIZMO': 'DarkMatter', 'RAMSES': 'DM', 'ART': 'darkmatter', 'CHANGA': 'DarkMatter'}
    dm_m = reg[(dm_name_dict[codetp],'particle_mass')].to('Msun').v.tolist()
    dm_coor = reg[(dm_name_dict[codetp],'particle_position')].to('kpc').v.tolist()
    dm_vel = reg[(dm_name_dict[codetp],'particle_velocity')].to('km/s').v.tolist()
    output_dm = {'mass':dm_m, 'coor':dm_coor, 'vel':dm_vel}
    np.save('./metadata/branch-0/dm_%s.npy' % index,output_dm)

index_i = 115
index_f = 134
codetp = 'GADGET3'
halotree_file = 'halotree_Thinh_structure_refinedboundary_2nd.npy'
extract_dm(index_i, codetp, halotree_file)
extract_dm(index_f, codetp, halotree_file)
    

