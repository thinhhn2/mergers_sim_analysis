import yt
from yt.data_objects.particle_filters import add_particle_filter
import numpy as np
import matplotlib.pyplot as plt
import sys, os

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

#-------------------------------------------------------------------------------------------
#LOAD DATA

tree_name = sys.argv[1] #for example 'halotree_Thinh_structure_with_com.npy'
code_name = sys.argv[2] #Select among ENZO, GADGET3, AREPO, GIZMO etc.
start_idx = int(sys.argv[3]) #if we want to start from a specific snapshot (when restarting, for example)
branch_name = sys.argv[4] 

tree = np.load(tree_name,allow_pickle=True).tolist()
pfs = np.loadtxt('pfs_manual.dat',dtype='str')
snapshot_idx = list(tree['0'].keys())[start_idx:]

#-------------------------------------------------------------------------------------------
#DEFINE FUNCTIONS

def stars(pfilter, data):
     filter_stars = np.logical_and(data["all", "particle_type"] == 2, data["all", "particle_mass"].to('Msun') > 1)
     return filter_stars

def find_scale_radius(s_distance_to_center,rvir, s_mass_each, percent_lim = 0.5):
    """
    This function calculates the stellar half-mass radius (or any other scale radius) of the galaxy, which is defined as the distance
    that encloses 50% (default) of the stellar mass in the Rvir region. 

    Note that this is the distance from the rotational axis, not the distance from the center of mass
    """
    bin_dist = np.linspace(0,1.5*rvir,500) #the 1.5 coefficient is to account when the reg.sphere does not load a perfect sphere
    mass_star_distance = []
    for i in range(1,len(bin_dist)):
        mass_star_bin_distance = np.sum(np.array(s_mass_each)[(s_distance_to_center>bin_dist[i-1]) & (s_distance_to_center<=bin_dist[i])])
        mass_star_distance.append(mass_star_bin_distance)

    mass_star_distance_cumsum = np.cumsum(mass_star_distance)

    mass_star_distance_cumsum_percent = mass_star_distance_cumsum/np.sum(s_mass_each)

    #Find the distance that begins to enclose 50% of the stellar mass in the Rvir region (i.e. scale distance)
    scale_radius = bin_dist[1:][mass_star_distance_cumsum_percent > percent_lim][0]

    return scale_radius

def Bullock_spin(spin_dist, dist_to_com, mass_each, angmoment_each):
    #THIS FUNCTION CALCULATES THE BULLOCK SPIN PARAMETER FOR BARYONIC MATTER
    #spin_dist: The distance where we evaluate the Bullock spin parameter

    spin = dist_to_com < spin_dist #this should be in spherical coordinates, not cylindrical
    spin_mass_each = np.array(mass_each)[spin]
    spin_mass = np.sum(spin_mass_each)

    spin_angmoment_each = angmoment_each[spin]
    spin_angmoment_total = np.sum(spin_angmoment_each,axis=0)

    spin_angmoment_magnitude = np.sum(spin_angmoment_total**2)**0.5

    #Calculate the Bullock Spin Parameter. Note that this is spin*sqrt(G)
    spin_param = spin_angmoment_magnitude/np.sqrt(2*spin_dist*spin_mass**3)
    return spin_param

#-------------------------------------------------------------------------------------------
#RUNNING MAIN CODE

my_storage = {}
for sto, idx in yt.parallel_objects(snapshot_idx, nprocs-1,storage = my_storage):

    if code_name == 'ENZO':
        ds = yt.load(pfs[int(idx)])
        add_particle_filter("stars", function=stars, filtered_type="all", requires=["particle_type","particle_mass"])
        ds.add_particle_filter("stars")
    
    if code_name == 'GADGET3' or code_name == 'AREPO':
        ds = yt.load(pfs[int(idx)],unit_base = {"length": (1.0, "Mpccm/h")})
    
    if code_name == 'GIZMO' or code_name == 'GEAR' or code_name == 'ART' or code_name == 'RAMSES' or code_name == 'CHANGA':
        ds = yt.load(pfs[int(idx)]) #GIZMO, GEAR, ART, and RAMSES automatically includes the correct conversion factor    

    star_data = np.load('metadata/%s/stars_%s.npy' % (branch_name,idx),allow_pickle=True).tolist()
    bary_data = np.load('metadata/%s/bary_%s.npy' % (branch_name,idx),allow_pickle=True).tolist()

    com_coor_bary = star_data['com_coor_bary']
    com_vel_bary = star_data['com_vel_bary']
    com_coor_star = star_data['com_coor_star']
    com_vel_star = star_data['com_vel_star']
    s_mass_each = star_data['mass']
    s_coor_each = star_data['coor']
    s_vel_each = star_data['vel']

    if s_mass_each == []: #if there is no star particle in the halo, then skip this snapshot
        continue
    
    s_rel_coor_each = s_coor_each - com_coor_bary
    s_rel_vel_each = s_vel_each - com_vel_bary

    #Calculate the stellar angular momentum
    s_relative_momentum_each = []
    for i in range(len(s_mass_each)):
        s_relative_momentum_each.append(s_mass_each[i]*s_rel_vel_each[i])

    s_relative_momentum_each = np.array(s_relative_momentum_each)
    
    s_angmoment_each = np.cross(s_rel_coor_each,s_relative_momentum_each)
    s_angmoment = np.sum(s_angmoment_each,axis=0)
    s_angmoment_unitvec = s_angmoment/np.sum(s_angmoment**2)**0.5

    #Calculate the baryonic angular momentum
    bary_rel_coor_each = bary_data['rel_coor']
    bary_rel_vel_each = bary_data['rel_vel']
    bary_mass_each = bary_data['mass']
    bary_rel_momentum_each = bary_data['rel_momentum']
    bary_angmoment_each = bary_data['angmoment']

    bary_angmoment = np.sum(bary_angmoment_each,axis=0)
    bary_angmoment_unitvec = bary_angmoment/np.sum(bary_angmoment**2)**0.5

    rvir_codelength = tree['0'][idx]['Rvir']*ds.units.code_length
    rvir = rvir_codelength.to('kpc').v.tolist()

    #Distance from the particles to the center of mass
    bary_dist_to_com = np.sqrt(np.sum(bary_rel_coor_each**2,axis=1))
    s_dist_to_com = np.sqrt(np.sum(s_rel_coor_each**2,axis=1))

    #Calculate the scale radius
    s_halfmass_radius = find_scale_radius(s_dist_to_com,rvir,s_mass_each, percent_lim=0.5)
    bary_halfmass_radius = find_scale_radius(bary_dist_to_com,rvir,bary_mass_each, percent_lim=0.5)
    s_90_radius = find_scale_radius(s_dist_to_com,rvir,s_mass_each, percent_lim=0.9)
    bary_90_radius = find_scale_radius(bary_dist_to_com,rvir,bary_mass_each, percent_lim=0.9)

    #-------------------------------------------------------------------------------------------
    #THIS SECTION CALCULATES THE BULLOCK SPIN PARAMETER. Note that this is spin*sqrt(G)
    s_spin_param_halfmass = Bullock_spin(s_halfmass_radius, s_dist_to_com, s_mass_each, s_angmoment_each)
    bary_spin_param_halfmass = Bullock_spin(bary_halfmass_radius, bary_dist_to_com, bary_mass_each, bary_angmoment_each)
    s_spin_param_90 = Bullock_spin(s_90_radius, s_dist_to_com, s_mass_each, s_angmoment_each)
    bary_spin_param_90 = Bullock_spin(bary_90_radius, bary_dist_to_com, bary_mass_each, bary_angmoment_each)
    s_spin_param_vir = Bullock_spin(rvir, s_dist_to_com, s_mass_each, s_angmoment_each)
    bary_spin_param_vir = Bullock_spin(rvir, bary_dist_to_com, bary_mass_each, bary_angmoment_each)

    output = {}
    output['s_halfmass_radius'] = s_halfmass_radius
    output['bary_halfmass_radius'] = bary_halfmass_radius
    output['s_90_radius'] = s_90_radius
    output['bary_90_radius'] = bary_90_radius
    output['s_spin_param_halfmass'] = s_spin_param_halfmass
    output['bary_spin_param_halfmass'] = bary_spin_param_halfmass
    output['s_spin_param_90'] = s_spin_param_90
    output['bary_spin_param_90'] = bary_spin_param_90
    output['s_spin_param_vir'] = s_spin_param_vir
    output['bary_spin_param_vir'] = bary_spin_param_vir

    np.save('metadata/%s/BullockSpin_%s.npy' % (branch_name,idx),output)







