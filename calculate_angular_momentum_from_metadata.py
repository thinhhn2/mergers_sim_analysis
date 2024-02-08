import numpy as np 
import glob as glob
import yt
import sys, os

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

branch_name = sys.argv[1]

bary_files = glob.glob('./metadata/%s/bary_*.npy' % branch_name)

my_storage = {}
for sto, file in yt.parallel_objects(bary_files, nprocs-1,storage = my_storage):

    bary_data = np.load(file,allow_pickle=True).tolist()
    bary_rel_coor_each = bary_data['rel_coor']
    bary_rel_vel_each = bary_data['rel_vel']
    bary_mass_each = bary_data['mass']
    bary_rel_momentum_each = bary_data['rel_momentum']
    bary_angmoment_each = bary_data['angmoment']
    com_coor_bary = bary_data['com_coor_bary']
    com_vel_bary = bary_data['com_vel_bary']

    bary_angmoment = np.sum(bary_angmoment_each,axis=0)
    bary_angmoment_unitvec = bary_angmoment/np.sum(bary_angmoment**2)**0.5
    bary_angmoment_magnitude = np.sum(bary_angmoment**2)**0.5
    bary_angmoment_specific = bary_angmoment_magnitude/np.sum(bary_mass_each)
    idx = file.split('_')[-1].split('.')[0] #obtain the number of out of the file name

    sto.result = {}
    sto.result['angmoment'] = bary_angmoment
    sto.result['angmoment_unitvec'] = bary_angmoment_unitvec
    sto.result['angmoment_magnitude'] = bary_angmoment_magnitude
    sto.result['angmoment_specific'] = bary_angmoment_specific
    sto.result['com_coor_bary'] = com_coor_bary
    sto.result['com_vel_bary'] = com_vel_bary
    sto.result['idx'] = idx

if yt.is_root():
    output = {}
    for c,vals in sorted(my_storage.items()):
        output[vals['idx']] = {}
        output[vals['idx']]['angmoment'] = vals['angmoment']
        output[vals['idx']]['angmoment_unitvec'] = vals['angmoment_unitvec']
        output[vals['idx']]['angmoment_magnitude'] = vals['angmoment_magnitude']
        output[vals['idx']]['angmoment_specific'] = vals['angmoment_specific']
        output[vals['idx']]['com_coor_bary'] = vals['com_coor_bary']
        output[vals['idx']]['com_vel_bary'] = vals['com_vel_bary']
    np.save('./metadata/%s/angmoment_%s.npy' % (branch_name,branch_name), output)

