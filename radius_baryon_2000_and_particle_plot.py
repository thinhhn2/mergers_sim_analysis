#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 18:29:51 2024

@author: thinhnguyen
"""
import numpy as np
import yt
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.units as u 
from yt.data_objects.particle_filters import add_particle_filter
from yt.data_objects.unions import ParticleUnion
from mpl_toolkits.axes_grid1 import AxesGrid
import os

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

def univDen(ds):
    # Hubble constant
    H0 = ds.hubble_constant * 100 * u.km/u.s/u.Mpc
    H = H0**2 * (ds.omega_matter*(1 + ds.current_redshift)**3 + ds.omega_lambda)  # Technically H^2
    G = 6.67e-11 * u.m**3/u.s**2/u.kg
    # Density of the universe
    den = (3*H/(8*np.pi*G)).to("kg/m**3") / u.kg * u.m**3
    return den.value

def Find_Com_and_virRad(initial_gal_com, halo_rvir, pos, mass, ds, oden=500):
    #Make sure the unit is in kg and m
    extension = 0.1 #default is 0.1
    virRad = 0.01*halo_rvir #default is 0.05
    com = initial_gal_com
    fden = np.inf
    while extension >= 0.01:
        virRad_new = virRad + extension*halo_rvir
        r = np.linalg.norm(pos - com, axis=1)
        pos_in = pos[r < virRad_new]
        mass_in = mass[r < virRad_new]
        den = np.sum(mass_in) / (4/3 * np.pi * virRad_new**3)
        uden = univDen(ds)
        fden = den/uden
        com_new = np.average(pos_in, weights=mass_in, axis=0)
        if fden < oden:
            extension -= 0.01
        else:
            com = com_new
            virRad = virRad_new
    return com, virRad


codetp = 'GEAR'
os.chdir('/scratch/bbvl/tnguyen2/%s' % codetp)
if not os.path.exists('/scratch/bbvl/tnguyen2/%s/radius_2000_%s' % (codetp, codetp)):
    os.mkdir('/scratch/bbvl/tnguyen2/%s/radius_2000_%s' % (codetp, codetp))

if codetp == 'ENZO':
    tree = np.load('/scratch/bbvl/tnguyen2/ENZO/halotree_Thinh_structure_with_com.npy',allow_pickle=True).tolist()
elif codetp == 'AREPO':
    tree = np.load('/scratch/bbvl/tnguyen2/AREPO/halotree_Thinh_structure_refinedboundary_2nd_manual_add_2nd_merger.npy',allow_pickle=True).tolist()
else:
    tree = np.load('/scratch/bbvl/tnguyen2/%s/halotree_Thinh_structure_refinedboundary_2nd.npy' % codetp, allow_pickle=True).tolist()

pfs = np.loadtxt('/scratch/bbvl/tnguyen2/%s/pfs_manual.dat' % codetp, dtype=str)

branch_idx = '0_10'

my_storage= {}

oden_lim = 10000 #R2000 (generally) and, R5000, R7500, or R10000 (to isolate the secondary better)
#for sto, snapshot_idx in yt.parallel_objects(list(tree[branch_idx].keys())[73:], nprocs-1, storage = my_storage):
for snapshot_idx in list(tree['0'].keys())[327:330]:
#for snapshot_idx in [list(tree['0'].keys())[129]]:
    #
    print('Starting the analysis for', snapshot_idx)
    if codetp == 'GADGET3' or codetp == 'AREPO':
        ds = yt.load(pfs[int(snapshot_idx)], unit_base = {"length": (1.0, "Mpccm/h")})
    else: 
        ds = yt.load(pfs[int(snapshot_idx)])
    #
    if codetp == 'ENZO':
        if branch_idx == '0_9' and int(snapshot_idx) > 141:
            halo = tree['0'][snapshot_idx]
            halo_com = halo['coor']
            halo_rvir = halo['Rvir']
        else:
            halo = tree[branch_idx][snapshot_idx]
            halo_com = halo['coor']
            halo_rvir = halo['Rvir']
    elif codetp == 'GADGET3':
        if branch_idx == '0_9' and int(snapshot_idx) > 133:
            halo = tree['0'][snapshot_idx]
            halo_com = halo['coor']
            halo_rvir = halo['Rvir']
        else:
            halo = tree[branch_idx][snapshot_idx]
            halo_com = halo['coor']
            halo_rvir = halo['Rvir']
    elif codetp == 'AREPO':
        if branch_idx == '0_6' and int(snapshot_idx) > 133:
            halo = tree['0'][snapshot_idx]
            halo_com = halo['coor']
            halo_rvir = halo['Rvir']
        else:
            halo = tree[branch_idx][snapshot_idx]
            halo_com = halo['coor']
            halo_rvir = halo['Rvir']
    elif codetp == 'GIZMO':
        if branch_idx == '0_2' and int(snapshot_idx) > 140:
            halo = tree['0'][snapshot_idx]
            halo_com = halo['coor']
            halo_rvir = halo['Rvir']
        else:
            halo = tree[branch_idx][snapshot_idx]
            halo_com = halo['coor']
            halo_rvir = halo['Rvir']
    elif codetp == 'CHANGA':
        if branch_idx == '0_5' and int(snapshot_idx) > 108:
            halo = tree['0'][snapshot_idx]
            halo_com = halo['coor']
            halo_rvir = halo['Rvir']
        else:
            halo = tree[branch_idx][snapshot_idx]
            halo_com = halo['coor']
            halo_rvir = halo['Rvir']
    elif codetp == 'RAMSES':
        if branch_idx == '0_4' and int(snapshot_idx) > 137:
            halo = tree['0'][snapshot_idx]
            halo_com = halo['coor']
            halo_rvir = halo['Rvir']
        else:
            halo = tree[branch_idx][snapshot_idx]
            halo_com = halo['coor']
            halo_rvir = halo['Rvir']
    elif codetp == 'GEAR':
        if branch_idx == '0_10' and int(snapshot_idx) > 394:
            halo = tree['0'][snapshot_idx]
            halo_com = halo['coor']
            halo_rvir = halo['Rvir']
        else:
            halo = tree[branch_idx][snapshot_idx]
            halo_com = halo['coor']
            halo_rvir = halo['Rvir']
    else:
        halo = tree[branch_idx][snapshot_idx]
        halo_com = halo['coor']
        halo_rvir = halo['Rvir']
    #
    #Finding R2000 using gas + stars particles
    if codetp == 'ENZO':
        if branch_idx == '0_9' and int(snapshot_idx) > 141:
            bary_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-0/bary_%s.npy' % (codetp, snapshot_idx), allow_pickle=True).tolist()
        else:
            bary_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-%s/bary_%s.npy' % (codetp, branch_idx, snapshot_idx), allow_pickle=True).tolist()
    elif codetp == 'GADGET3':
        if branch_idx == '0_9' and int(snapshot_idx) > 133:
            bary_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-0/bary_%s.npy' % (codetp, snapshot_idx), allow_pickle=True).tolist()
        else:
            bary_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-%s/bary_%s.npy' % (codetp, branch_idx, snapshot_idx), allow_pickle=True).tolist()
    elif codetp == 'AREPO':
        if branch_idx == '0_6' and int(snapshot_idx) > 133:
            bary_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-0/bary_%s.npy' % (codetp, snapshot_idx), allow_pickle=True).tolist()
        else:
            bary_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-%s/bary_%s.npy' % (codetp, branch_idx, snapshot_idx), allow_pickle=True).tolist()
    elif codetp == 'GIZMO':
        if branch_idx == '0_2' and int(snapshot_idx) > 140:
            bary_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-0/bary_%s.npy' % (codetp, snapshot_idx), allow_pickle=True).tolist()
        else:
            bary_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-%s/bary_%s.npy' % (codetp, branch_idx, snapshot_idx), allow_pickle=True).tolist()
    elif codetp == 'CHANGA':
        if branch_idx == '0_5' and int(snapshot_idx) > 108:
            bary_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-0/bary_%s.npy' % (codetp, snapshot_idx), allow_pickle=True).tolist()
        else:
            bary_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-%s/bary_%s.npy' % (codetp, branch_idx, snapshot_idx), allow_pickle=True).tolist()
    elif codetp == 'RAMSES':
        if branch_idx == '0_4' and int(snapshot_idx) > 137:
            bary_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-0/bary_%s.npy' % (codetp, snapshot_idx), allow_pickle=True).tolist()
        else:
            bary_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-%s/bary_%s.npy' % (codetp, branch_idx, snapshot_idx), allow_pickle=True).tolist()
    elif codetp == 'GEAR':
        if branch_idx == '0_10' and int(snapshot_idx) > 394:
            bary_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-0/bary_%s.npy' % (codetp, snapshot_idx), allow_pickle=True).tolist()
        else:
            bary_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-%s/bary_%s.npy' % (codetp, branch_idx, snapshot_idx), allow_pickle=True).tolist()
    #
    pos_kpc = bary_metadata['rel_coor'] + bary_metadata['com_coor_bary']
    pos = (pos_kpc*ds.units.kpc).to('m').v
    mass = (bary_metadata['mass']*ds.units.Msun).to('kg').v
    #ART ALSO HAS PROBLEMS FROM SNAPSHOT IDX 122 TO 131, NEED TO ADD THE INITIAL COM MANUALLY
    if codetp == 'ENZO':
        if branch_idx == '0':
            if int(snapshot_idx) == 143:
                initial_gal_com = np.array([0.49442,0.51790,0.50258])
            elif int(snapshot_idx) == 144:
                initial_gal_com = np.array([0.49445,0.51805,0.50263])
            elif int(snapshot_idx) == 145:
                initial_gal_com = np.array([0.49439,0.51815,0.50270])
            elif int(snapshot_idx) == 146:
                initial_gal_com = np.array([0.49429,0.51840,0.50277])
            elif int(snapshot_idx) == 147:
                initial_gal_com = np.array([0.49419,0.51855,0.50285])
            elif int(snapshot_idx) == 148:
                initial_gal_com = np.array([0.49405,0.51869,0.50297])
            elif int(snapshot_idx) == 149:
                initial_gal_com = np.array([0.49393,0.51885,0.50312])
            elif int(snapshot_idx) == 150:
                initial_gal_com = np.array([0.49380,0.51900,0.50332])
            elif int(snapshot_idx) == 151:
                initial_gal_com = np.array([0.49372,0.51917,0.50339])
            else:
                initial_gal_com = np.array(halo_com)
        elif branch_idx == '0_9': #first merger in ENZO
            if int(snapshot_idx) == 142: #do manually for 142 to 148 (at 149, the two galaxies fall in the same R2000 radius)
                initial_gal_com = np.array([0.49425,0.51780,0.50286])
            elif int(snapshot_idx) == 143: 
                initial_gal_com = np.array([0.49412,0.51793,0.50302])
            elif int(snapshot_idx) == 144: 
                initial_gal_com = np.array([0.49402,0.51808,0.50312])
            elif int(snapshot_idx) == 145: 
                initial_gal_com = np.array([0.49396,0.51826,0.50321])
            elif int(snapshot_idx) == 146: 
                initial_gal_com = np.array([0.49390,0.51841,0.50327])
            elif int(snapshot_idx) == 147: 
                initial_gal_com = np.array([0.49387,0.51858,0.50331])
            elif int(snapshot_idx) == 148: 
                initial_gal_com = np.array([0.49384,0.51871,0.50331])
            elif int(snapshot_idx) == 149: 
                initial_gal_com = np.array([0.49383,0.51885,0.50330])
            else:
                initial_gal_com = np.array(halo_com)
    elif codetp == 'GADGET3':
        if branch_idx == '0':
            initial_gal_com = np.array(halo_com)
        elif branch_idx == '0_9':
            if int(snapshot_idx) == 134: #do manually for 134 to 141 
                initial_gal_com = np.array([29.634, 31.090, 30.169])
            elif int(snapshot_idx) == 135: 
                initial_gal_com = np.array([29.626, 31.095, 30.177])
            elif int(snapshot_idx) == 136: 
                initial_gal_com = np.array([29.621, 31.103, 30.180])
            elif int(snapshot_idx) == 137: 
                initial_gal_com = np.array([29.618, 31.112, 30.182])
            elif int(snapshot_idx) == 138: 
                initial_gal_com = np.array([29.619, 31.124, 30.181]) #Try R10000
            elif int(snapshot_idx) == 139: 
                initial_gal_com = np.array([29.622, 31.137, 30.180])
            elif int(snapshot_idx) == 140: 
                initial_gal_com = np.array([29.623, 31.159, 30.183]) #Try R10000
            elif int(snapshot_idx) == 141: 
                initial_gal_com = np.array([29.616, 31.168, 30.198])
            else:
                initial_gal_com = np.array(halo_com)
    elif codetp == 'AREPO':
        if branch_idx == '0':
            initial_gal_com = np.array(halo_com)
        elif branch_idx == '0_6':
            if int(snapshot_idx) == 134: #do manually for 134 to 141 
                initial_gal_com = np.array([29.625, 31.102, 30.175])
            elif int(snapshot_idx) == 135: 
                initial_gal_com = np.array([29.616, 31.111, 30.182])
            elif int(snapshot_idx) == 136: 
                initial_gal_com = np.array([29.612, 31.119, 30.187])
            elif int(snapshot_idx) == 137: 
                initial_gal_com = np.array([29.607, 31.129, 30.192])
            elif int(snapshot_idx) == 138: 
                initial_gal_com = np.array([29.605, 31.138, 30.192])
            elif int(snapshot_idx) == 139: 
                initial_gal_com = np.array([29.605, 31.147, 30.191])
            elif int(snapshot_idx) == 140: 
                initial_gal_com = np.array([29.605, 31.153, 30.192])
            elif int(snapshot_idx) == 141: 
                initial_gal_com = np.array([29.60, 31.165, 30.200])
            else:
                initial_gal_com = np.array(halo_com)
    elif codetp == 'GIZMO':
        if branch_idx == '0':
            initial_gal_com = np.array(halo_com)
        elif branch_idx == '0_2':
            if int(snapshot_idx) == 141: #do manually for 141 to 145
                initial_gal_com = np.array([29638, 31086, 30164])
            elif int(snapshot_idx) == 142: 
                initial_gal_com = np.array([29632, 31091, 30168])
            elif int(snapshot_idx) == 143: 
                initial_gal_com = np.array([29630, 31101, 30170])
            elif int(snapshot_idx) == 144: 
                initial_gal_com = np.array([29630, 31115, 30168])
            elif int(snapshot_idx) == 145: 
                initial_gal_com = np.array([29623, 31125, 30170])
            else:
                initial_gal_com = np.array(halo_com)
    elif codetp == 'CHANGA':
        if branch_idx == '0':
            if int(snapshot_idx) == 112:
                initial_gal_com = np.array([-0.00605,0.01862,0.00256])
            if int(snapshot_idx) == 113:
                initial_gal_com = np.array([-0.00608,0.01875,0.00259])
            if int(snapshot_idx) == 114:
                initial_gal_com = np.array([-0.00612,0.01884,0.00261])
            if int(snapshot_idx) == 116:
                initial_gal_com = np.array([-0.00625,0.01905,0.00269])
            else:
                initial_gal_com = np.array(halo_com)
        elif branch_idx == '0_5':
            if int(snapshot_idx) == 109: #do manually for 109 to 120
                initial_gal_com = np.array([-0.00614, 0.01836, 0.00281])
            elif int(snapshot_idx) == 110: 
                initial_gal_com = np.array([-0.00623, 0.01847, 0.00292])
            elif int(snapshot_idx) == 111: 
                initial_gal_com = np.array([-0.00630, 0.01855, 0.00300])
            elif int(snapshot_idx) == 112: 
                initial_gal_com = np.array([-0.00636, 0.01868, 0.00308])
            elif int(snapshot_idx) == 113: 
                initial_gal_com = np.array([-0.00640, 0.01880, 0.00311])
            elif int(snapshot_idx) == 114: 
                initial_gal_com = np.array([-0.00641, 0.01890, 0.00312])
            elif int(snapshot_idx) == 115: 
                initial_gal_com = np.array([-0.00645, 0.01900, 0.00315])
            elif int(snapshot_idx) == 116: 
                initial_gal_com = np.array([-0.00645, 0.01910, 0.00317])
            elif int(snapshot_idx) == 117: 
                initial_gal_com = np.array([-0.00646, 0.01920, 0.00314])
            elif int(snapshot_idx) == 118: 
                initial_gal_com = np.array([-0.00649, 0.01929, 0.00312])
            elif int(snapshot_idx) == 119: 
                initial_gal_com = np.array([-0.00647, 0.01952, 0.00305])
            elif int(snapshot_idx) == 120: 
                initial_gal_com = np.array([-0.00669, 0.01972, 0.00333])
            else:
                initial_gal_com = np.array(halo_com)
    elif codetp == 'RAMSES':
        if branch_idx == '0':
            initial_gal_com = np.array(halo_com)
        elif branch_idx == '0_4':
            if int(snapshot_idx) == 138: #do manually for 141 to 145
                initial_gal_com = np.array([0.49482, 0.51685, 0.50160])
            elif int(snapshot_idx) == 139: 
                initial_gal_com = np.array([0.49474, 0.51705, 0.50180])
            elif int(snapshot_idx) == 140: 
                initial_gal_com = np.array([0.49460, 0.51730, 0.50205])
            elif int(snapshot_idx) == 141: 
                initial_gal_com = np.array([0.49442, 0.51752, 0.50228])
            elif int(snapshot_idx) == 142: 
                initial_gal_com = np.array([0.49420, 0.51765, 0.50251])
            elif int(snapshot_idx) == 143: 
                initial_gal_com = np.array([0.49402, 0.51775, 0.50270])
            elif int(snapshot_idx) == 144: 
                initial_gal_com = np.array([0.49385, 0.51785, 0.50282])
            elif int(snapshot_idx) == 145: 
                initial_gal_com = np.array([0.49375, 0.51800, 0.50293])
            elif int(snapshot_idx) == 146: 
                initial_gal_com = np.array([0.49365, 0.51813, 0.50302])
            elif int(snapshot_idx) == 147: 
                initial_gal_com = np.array([0.49360, 0.51832, 0.50310])
            elif int(snapshot_idx) == 148: 
                initial_gal_com = np.array([0.49358, 0.51850, 0.50314])
            elif int(snapshot_idx) == 149: 
                initial_gal_com = np.array([0.49355, 0.51868, 0.50315])
            elif int(snapshot_idx) == 150: 
                initial_gal_com = np.array([0.49355, 0.51882, 0.50315])
            elif int(snapshot_idx) == 151: 
                initial_gal_com = np.array([0.49355, 0.51900, 0.50312])
            elif int(snapshot_idx) == 152: 
                initial_gal_com = np.array([0.49360, 0.51905, 0.50308])
            elif int(snapshot_idx) == 153: 
                initial_gal_com = np.array([0.49363, 0.51915, 0.50300])
            elif int(snapshot_idx) == 154: 
                initial_gal_com = np.array([0.49359, 0.51935, 0.50300])
            else:
                initial_gal_com = np.array(halo_com)
    elif codetp == 'GEAR':
        if branch_idx == '0':
            initial_gal_com = np.array(halo_com)
        elif branch_idx == '0_10':
            if int(snapshot_idx) == 395: #do manually for 141 to 145
                initial_gal_com = np.array([29600, 31152, 30193])
            elif int(snapshot_idx) == 396: 
                initial_gal_com = np.array([29600, 31155, 30193])
            elif int(snapshot_idx) == 397: 
                initial_gal_com = np.array([29612, 31157, 30190])
            elif int(snapshot_idx) == 377: 
                initial_gal_com = np.array([29641, 31098, 30153])
            elif int(snapshot_idx) == 378: 
                initial_gal_com = np.array([29637, 31103, 30159])
            elif int(snapshot_idx) == 379: 
                initial_gal_com = np.array([29631, 31107, 30165])
            elif int(snapshot_idx) == 380: 
                initial_gal_com = np.array([29627, 31110, 30170])
            else:
                initial_gal_com = np.array(halo_com)
    #
    initial_gal_com_m = (initial_gal_com*ds.units.code_length).in_units('m').v
    initial_gal_com_kpc = (initial_gal_com*ds.units.code_length).in_units('kpc').v
    halo_rvir_m = (halo_rvir*ds.units.code_length).to('m').v.tolist()
    halo_rvir_kpc = (halo_rvir*ds.units.code_length).in_units('kpc').v.tolist()
    gal_com_m, gal_r2000_m = Find_Com_and_virRad(initial_gal_com_m, halo_rvir_m, pos, mass, ds, oden=oden_lim) #R2000 using star and gas, for close interaction, may need to increase to R10000 to better isolate the secondary galaxy
    gal_com_kpc = (gal_com_m*ds.units.m).in_units('kpc').v
    gal_com = (gal_com_m*ds.units.m).in_units('code_length').v
    gal_r2000_kpc = (gal_r2000_m*ds.units.m).in_units('kpc').v.tolist()
    gal_r2000 = (gal_r2000_m*ds.units.m).in_units('code_length').v.tolist()
    print('The R2000/Rvir ratio is', gal_r2000_kpc/halo_rvir_kpc)
    #Export the data
    output = {}
    output['gal_com'] = gal_com
    output['gal_r2000'] = gal_r2000
    if oden_lim == 2000:
        np.save('radius_2000_%s/branch-%s/gal_com_r2000_SnapIdx_%s.npy' % (codetp, branch_idx, snapshot_idx), output)
    elif oden_lim == 5000:
        np.save('radius_2000_%s/branch-%s/gal_com_r5000_SnapIdx_%s.npy' % (codetp, branch_idx, snapshot_idx), output)
    elif oden_lim == 7500:
        np.save('radius_2000_%s/branch-%s/gal_com_r7500_SnapIdx_%s.npy' % (codetp, branch_idx, snapshot_idx), output)
    elif oden_lim == 10000:
        np.save('radius_2000_%s/branch-%s/gal_com_r10000_SnapIdx_%s.npy' % (codetp, branch_idx, snapshot_idx), output)
    #-----------------------------------------------------------------------------------------------
    #Plotting the surface mass density in 3 axis
    if codetp == 'ENZO':
        if branch_idx == '0_9' and int(snapshot_idx) > 141:
            star_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-0/stars_%s.npy' % (codetp, snapshot_idx), allow_pickle=True).tolist()
        else:
            star_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-%s/stars_%s.npy' % (codetp, branch_idx, snapshot_idx), allow_pickle=True).tolist()
    if codetp == 'GADGET3':
        if branch_idx == '0_9' and int(snapshot_idx) > 133:
            star_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-0/stars_%s.npy' % (codetp, snapshot_idx), allow_pickle=True).tolist()
        else:
            star_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-%s/stars_%s.npy' % (codetp, branch_idx, snapshot_idx), allow_pickle=True).tolist()
    if codetp == 'AREPO':
        if branch_idx == '0_6' and int(snapshot_idx) > 133:
            star_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-0/stars_%s.npy' % (codetp, snapshot_idx), allow_pickle=True).tolist()
        else:
            star_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-%s/stars_%s.npy' % (codetp, branch_idx, snapshot_idx), allow_pickle=True).tolist()
    if codetp == 'GIZMO':
        if branch_idx == '0_2' and int(snapshot_idx) > 140:
            star_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-0/stars_%s.npy' % (codetp, snapshot_idx), allow_pickle=True).tolist()
        else:
            star_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-%s/stars_%s.npy' % (codetp, branch_idx, snapshot_idx), allow_pickle=True).tolist()
    if codetp == 'CHANGA':
        if branch_idx == '0_5' and int(snapshot_idx) > 108:
            star_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-0/stars_%s.npy' % (codetp, snapshot_idx), allow_pickle=True).tolist()
        else:
            star_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-%s/stars_%s.npy' % (codetp, branch_idx, snapshot_idx), allow_pickle=True).tolist()
    if codetp == 'RAMSES':
        if branch_idx == '0_4' and int(snapshot_idx) > 137:
            star_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-0/stars_%s.npy' % (codetp, snapshot_idx), allow_pickle=True).tolist()
        else:
            star_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-%s/stars_%s.npy' % (codetp, branch_idx, snapshot_idx), allow_pickle=True).tolist()
    if codetp == 'GEAR':
        if branch_idx == '0_10' and int(snapshot_idx) > 394:
            star_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-0/stars_%s.npy' % (codetp, snapshot_idx), allow_pickle=True).tolist()
        else:
            star_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-%s/stars_%s.npy' % (codetp, branch_idx, snapshot_idx), allow_pickle=True).tolist()
    #
    pos_star_kpc = np.array(star_metadata['coor']) #in unit of kpc
    plt.figure(figsize=(15,5))
    #
    rel_pos_star_kpc = pos_star_kpc - gal_com_kpc
    rel_x = rel_pos_star_kpc[:,0]
    rel_y = rel_pos_star_kpc[:,1]
    rel_z = rel_pos_star_kpc[:,2]
    #
    ax1 = plt.subplot(1,3,1)
    _ = ax1.hist2d(rel_x, rel_y, bins=800, norm=mpl.colors.LogNorm(), cmap=mpl.cm.viridis)
    circle1 = plt.Circle((0, 0), gal_r2000_kpc, fill=False)
    ax1.add_patch(circle1)
    ax1.set_xlabel('x (kpc)', fontsize=16)
    ax1.set_ylabel('y (kpc)', fontsize=16)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    coor_lim = max(np.max(abs(rel_x)), np.max(abs(rel_y)))
    ax1.set_xlim(-coor_lim, coor_lim)
    ax1.set_ylim(-coor_lim, coor_lim)
    #
    ax2 = plt.subplot(1,3,2)
    _ = ax2.hist2d(rel_y, rel_z, bins=800, norm=mpl.colors.LogNorm(), cmap=mpl.cm.viridis)
    circle2 = plt.Circle((0, 0), gal_r2000_kpc, fill=False)
    ax2.add_patch(circle2)
    ax2.set_xlabel('y (kpc)', fontsize=16)
    ax2.set_ylabel('z (kpc)', fontsize=16)
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    coor_lim = max(np.max(abs(rel_y)), np.max(abs(rel_z)))
    ax2.set_xlim(-coor_lim, coor_lim)
    ax2.set_ylim(-coor_lim, coor_lim)
    if oden_lim == 2000:
        ax2.set_title(r'$r_{baryon,2000}$ = %.2f kpc = %.2f Rvir' % (gal_r2000_kpc, gal_r2000_kpc/halo_rvir_kpc), fontsize=16)
    elif oden_lim == 5000:
        ax2.set_title(r'$r_{baryon,5000}$ = %.2f kpc = %.2f Rvir' % (gal_r2000_kpc, gal_r2000_kpc/halo_rvir_kpc), fontsize=16)
    elif oden_lim == 7500:
        ax2.set_title(r'$r_{baryon,7500}$ = %.2f kpc = %.2f Rvir' % (gal_r2000_kpc, gal_r2000_kpc/halo_rvir_kpc), fontsize=16)
    elif oden_lim == 10000:
        ax2.set_title(r'$r_{baryon,10000}$ = %.2f kpc = %.2f Rvir' % (gal_r2000_kpc, gal_r2000_kpc/halo_rvir_kpc), fontsize=16)
    #
    ax3 = plt.subplot(1,3,3)
    _ = ax3.hist2d(rel_z, rel_x, bins=800, norm=mpl.colors.LogNorm(), cmap=mpl.cm.viridis)
    circle3 = plt.Circle((0, 0), gal_r2000_kpc, fill=False)
    ax3.add_patch(circle3)
    ax3.set_xlabel('z (kpc)', fontsize=16)
    ax3.set_ylabel('x (kpc)', fontsize=16)
    ax3.tick_params(axis='x', labelsize=16)
    ax3.tick_params(axis='y', labelsize=16)
    coor_lim = max(np.max(abs(rel_z)), np.max(abs(rel_x)))
    ax3.set_xlim(-coor_lim, coor_lim)
    ax3.set_ylim(-coor_lim, coor_lim)
    #
    plt.tight_layout()
    if oden_lim == 2000:
        plt.savefig("radius_2000_%s/branch-%s/surface_mass_density_%s_%s.png" % (codetp, branch_idx, branch_idx, snapshot_idx),dpi=300,bbox_inches='tight') 
    elif oden_lim == 5000:
        plt.savefig("radius_2000_%s/branch-%s/surface_mass_density_%s_%s_R5000.png" % (codetp, branch_idx, branch_idx, snapshot_idx),dpi=300,bbox_inches='tight') 
    elif oden_lim == 7500:
        plt.savefig("radius_2000_%s/branch-%s/surface_mass_density_%s_%s_R7500.png" % (codetp, branch_idx, branch_idx, snapshot_idx),dpi=300,bbox_inches='tight') 
    elif oden_lim == 10000:
        plt.savefig("radius_2000_%s/branch-%s/surface_mass_density_%s_%s_R10000.png" % (codetp, branch_idx, branch_idx, snapshot_idx),dpi=300,bbox_inches='tight')    
    plt.close()
    
#------------------------------------------------------------------------------
#Combine all the individual r2000 files into one
gal_com_r2000 = {}
snapidx_range = np.arange(123, 149+1, 1)
for snapidx in snapidx_range:
    data = np.load('gal_com_r2000_SnapIdx_%s.npy' % snapidx, allow_pickle=True).tolist()
    gal_com_r2000[snapidx] = [data['gal_com'], data['gal_r2000']]

np.save('gal_com_r2000_%s.npy' % branch_idx, gal_com_r2000)    