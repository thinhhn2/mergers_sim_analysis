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
    extension = 0.1
    virRad = 0.05*halo_rvir
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


codetp = 'ENZO'
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

branch_idx = '0'

my_storage= {}

for sto, snapshot_idx in yt.parallel_objects(list(tree[branch_idx].keys())[73:], nprocs-1, storage = my_storage):
    #
    print('Starting the analysis for', snapshot_idx)
    halo = tree[branch_idx][snapshot_idx]
    if codetp == 'GADGET3' or codetp == 'AREPO':
        ds = yt.load(pfs[int(snapshot_idx)], unit_base = {"length": (1.0, "Mpccm/h")})
    else: 
        ds = yt.load(pfs[int(snapshot_idx)])
    #
    halo_com = halo['coor']
    halo_rvir = halo['Rvir']
    #
    #Finding R2000 using gas + stars particles
    bary_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-0/bary_%s.npy' % (codetp, snapshot_idx), allow_pickle=True).tolist()
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
        if branch_idx == '0_9': #first merger in ENZO
            if int(snapshot_idx) == 143: #do manually for 143 to 148 (at 149, the two galaxies fall in the same R2000 radius)
                initial_gal_com = np.array([0.49412,0.51791,0.50302])
    if codetp == 'CHANGA':
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
    #
    initial_gal_com_m = (initial_gal_com*ds.units.code_length).in_units('m').v
    initial_gal_com_kpc = (initial_gal_com*ds.units.code_length).in_units('kpc').v
    halo_rvir_m = (halo_rvir*ds.units.code_length).to('m').v.tolist()
    halo_rvir_kpc = (halo_rvir*ds.units.code_length).in_units('kpc').v.tolist()
    gal_com_m, gal_r2000_m = Find_Com_and_virRad(initial_gal_com_m, halo_rvir_m, pos, mass, ds, oden=2000) #R2000 using star and gas
    gal_com_kpc = (gal_com_m*ds.units.m).in_units('kpc').v
    gal_com = (gal_com_m*ds.units.m).in_units('code_length').v
    gal_r2000_kpc = (gal_r2000_m*ds.units.m).in_units('kpc').v.tolist()
    gal_r2000 = (gal_r2000_m*ds.units.m).in_units('code_length').v.tolist()
    #Export the data
    output = {}
    output['gal_com'] = gal_com
    output['gal_r2000'] = gal_r2000
    np.save('radius_2000_%s/gal_com_r2000_SnapIdx_%s.npy' % (codetp, snapshot_idx), output)
    #-----------------------------------------------------------------------------------------------
    #Plotting the surface mass density in 3 axis
    star_metadata = np.load('/scratch/bbvl/tnguyen2/%s/metadata/branch-0/stars_%s.npy' % (codetp, snapshot_idx), allow_pickle=True).tolist()
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
    ax2.set_title(r'$r_{baryon,2000}$ = %.2f kpc = %.2f Rvir' % (gal_r2000_kpc, gal_r2000_kpc/halo_rvir_kpc), fontsize=16)
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
    plt.savefig("radius_2000_%s/surface_mass_density_%s_%s.png" % (codetp, branch_idx, snapshot_idx),dpi=300,bbox_inches='tight')    
    
#------------------------------------------------------------------------------
#Combine all the individual r2000 files into one
gal_com_r2000 = {}
snapidx_range = np.arange(98, 332, 1)
for snapidx in snapidx_range:
    data = np.load('gal_com_r2000_SnapIdx_%s.npy' % snapidx, allow_pickle=True).tolist()
    gal_com_r2000[snapidx] = [data['gal_com'], data['gal_r2000']]

np.save('gal_com_r2000.npy', gal_com_r2000)    