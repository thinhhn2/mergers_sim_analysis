import yt
import numpy as np
from yt.data_objects.particle_filters import add_particle_filter
import matplotlib.pyplot as plt
import sys, os

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

#-------------------------------------------------------------------------------------------
#LOAD DATA
def stars(pfilter, data):
    filter_stars = np.logical_and(data["all", "particle_type"] == 2, data["all", "particle_mass"].to('Msun') > 1)
    return filter_stars

tree_name = sys.argv[1] #for example 'halotree_Thinh_structure_with_com.npy'
code_name = sys.argv[2] #Select among ENZO, GADGET3, AREPO, GIZMO, GEAR, RAMSES, and ART
start_idx = int(sys.argv[3]) #if we want to start from a specific snapshot (when restarting, for example)

tree = np.load(tree_name,allow_pickle=True).tolist()
pfs = np.loadtxt('pfs_manual.dat',dtype='str')
snapshot_idx = list(tree['0'].keys())[start_idx:]
if yt.is_root():
    os.mkdir('./metadata')
#-------------------------------------------------------------------------------------------
#MAIN CODE

my_storage = {}
for sto, idx in yt.parallel_objects(snapshot_idx, nprocs-1,storage = my_storage):
    if code_name == 'ENZO':
        ds = yt.load(pfs[int(idx)])

        coor = tree['0'][idx]['coor']
        rvir = tree['0'][idx]['Rvir']

        reg = ds.sphere(coor,(rvir,'code_length'))

        add_particle_filter("stars", function=stars, filtered_type="all", requires=["particle_type","particle_mass"])
        ds.add_particle_filter("stars")

        com_coor_star = reg.quantities.center_of_mass(use_gas = False, use_particles = True, particle_type='stars').to('kpc').v
        com_vel_star = reg.quantities.bulk_velocity(use_gas = False, use_particles = True, particle_type='stars').to('km/s').v

        com_coor_bary = reg.quantities.center_of_mass(use_gas = True, use_particles = True, particle_type='stars').to('kpc').v
        com_vel_bary = reg.quantities.bulk_velocity(use_gas = True, use_particles = True, particle_type='stars').to('km/s').v

        #Calculating stars' metadata
        s_mass_each = reg[("stars", "particle_mass")].in_units("Msun").v.tolist()
        s_coor_each = reg[("stars", "particle_position")].in_units("kpc").v.tolist()
        s_vel_each = reg[("stars", "particle_velocity")].in_units("km/s").v.tolist()

        #Calculating gas' metadata
        g_mass_each = reg[("gas","cell_mass")].in_units("Msun").v.tolist()
        g_x_each = reg[("gas","x")].in_units("kpc").v.tolist()
        g_y_each = reg[("gas","y")].in_units("kpc").v.tolist()
        g_z_each = reg[("gas","z")].in_units("kpc").v.tolist()
        g_velx_each = reg[("gas","velocity_x")].in_units("km/s").v.tolist()
        g_vely_each = reg[("gas","velocity_y")].in_units("km/s").v.tolist()
        g_velz_each = reg[("gas","velocity_z")].in_units("km/s").v.tolist()
        g_coor_each = []
        g_vel_each = []
        for i in range(len(g_x_each)):
            g_coor_each.append([g_x_each[i],g_y_each[i],g_z_each[i]])
            g_vel_each.append([g_velx_each[i],g_vely_each[i],g_velz_each[i]])
        g_coor_each = np.array(g_coor_each)
        g_vel_each = np.array(g_vel_each)

    if code_name == 'GADGET3' or code_name == 'AREPO':
        ds = yt.load(pfs[int(idx)],unit_base = {"length": (1.0, "Mpccm/h")})

        coor = tree['0'][idx]['coor']
        rvir = tree['0'][idx]['Rvir']

        reg = ds.sphere(coor,(rvir,'code_length'))

        com_coor_star = reg.quantities.center_of_mass(use_gas = False, use_particles = True, particle_type='PartType4').to('kpc').v
        com_vel_star = reg.quantities.bulk_velocity(use_gas = False, use_particles = True, particle_type='PartType4').to('km/s').v

        com_coor_bary = reg.quantities.center_of_mass(use_gas = True, use_particles = True, particle_type='PartType4').to('kpc').v
        com_vel_bary = reg.quantities.bulk_velocity(use_gas = True, use_particles = True, particle_type='PartType4').to('km/s').v

        #Calculating stars' metadata
        s_mass_each = reg[("PartType4", "particle_mass")].in_units("Msun").v.tolist()
        s_coor_each = reg[("PartType4", "particle_position")].in_units("kpc").v.tolist()
        s_vel_each = reg[("PartType4", "particle_velocity")].in_units("km/s").v.tolist()

        #Calculating gas' metadata
        g_mass_each = reg[("PartType0", "particle_mass")].in_units("Msun").v.tolist()
        g_coor_each = reg[("PartType0", "particle_position")].in_units("kpc").v.tolist()
        g_vel_each = reg[("PartType0", "particle_velocity")].in_units("km/s").v.tolist()
    
    if code_name == 'GIZMO':
        ds = yt.load(pfs[int(idx)]) #GIZMO automatically includes the correct conversion factor

        coor = tree['0'][idx]['coor']
        rvir = tree['0'][idx]['Rvir']

        reg = ds.sphere(coor,(rvir,'code_length'))

        com_coor_star = reg.quantities.center_of_mass(use_gas = False, use_particles = True, particle_type='PartType4').to('kpc').v
        com_vel_star = reg.quantities.bulk_velocity(use_gas = False, use_particles = True, particle_type='PartType4').to('km/s').v

        com_coor_bary = reg.quantities.center_of_mass(use_gas = True, use_particles = True, particle_type='PartType4').to('kpc').v
        com_vel_bary = reg.quantities.bulk_velocity(use_gas = True, use_particles = True, particle_type='PartType4').to('km/s').v

        #Calculating stars' metadata
        s_mass_each = reg[("PartType4", "particle_mass")].in_units("Msun").v.tolist()
        s_coor_each = reg[("PartType4", "particle_position")].in_units("kpc").v.tolist()
        s_vel_each = reg[("PartType4", "particle_velocity")].in_units("km/s").v.tolist()

        #Calculating gas' metadata
        g_mass_each = reg[("PartType0", "particle_mass")].in_units("Msun").v.tolist()
        g_coor_each = reg[("PartType0", "particle_position")].in_units("kpc").v.tolist()
        g_vel_each = reg[("PartType0", "particle_velocity")].in_units("km/s").v.tolist()

    if code_name == 'GEAR':
        ds = yt.load(pfs[int(idx)]) #GIZMO automatically includes the correct conversion factor

        coor = tree['0'][idx]['coor']
        rvir = tree['0'][idx]['Rvir']

        reg = ds.sphere(coor,(rvir,'code_length'))

        com_coor_star = reg.quantities.center_of_mass(use_gas = False, use_particles = True, particle_type='PartType1').to('kpc').v
        com_vel_star = reg.quantities.bulk_velocity(use_gas = False, use_particles = True, particle_type='PartType1').to('km/s').v

        com_coor_bary = reg.quantities.center_of_mass(use_gas = True, use_particles = True, particle_type='PartType1').to('kpc').v
        com_vel_bary = reg.quantities.bulk_velocity(use_gas = True, use_particles = True, particle_type='PartType1').to('km/s').v

        #Calculating stars' metadata
        s_mass_each = reg[("PartType1", "particle_mass")].in_units("Msun").v.tolist()
        s_coor_each = reg[("PartType1", "particle_position")].in_units("kpc").v.tolist()
        s_vel_each = reg[("PartType1", "particle_velocity")].in_units("km/s").v.tolist()

        #Calculating gas' metadata
        g_mass_each = reg[("PartType0", "particle_mass")].in_units("Msun").v.tolist()
        g_coor_each = reg[("PartType0", "particle_position")].in_units("kpc").v.tolist()
        g_vel_each = reg[("PartType0", "particle_velocity")].in_units("km/s").v.tolist()
    

    if code_name == 'ART':
        ds = yt.load(pfs[int(idx)]) #ART automatically includes the correct conversion factor

        coor = tree['0'][idx]['coor']
        rvir = tree['0'][idx]['Rvir']

        reg = ds.sphere(coor,(rvir,'code_length'))

        com_coor_star = reg.quantities.center_of_mass(use_gas = False, use_particles = True, particle_type='PartType1').to('kpc').v
        com_vel_star = reg.quantities.bulk_velocity(use_gas = False, use_particles = True, particle_type='PartType1').to('km/s').v

        com_coor_bary = reg.quantities.center_of_mass(use_gas = True, use_particles = True, particle_type='PartType1').to('kpc').v
        com_vel_bary = reg.quantities.bulk_velocity(use_gas = True, use_particles = True, particle_type='PartType1').to('km/s').v

        #Calculating stars' metadata
        s_mass_each = reg[("stars", "particle_mass")].in_units("Msun").v.tolist()
        s_coor_each = reg[("stars", "particle_position")].in_units("kpc").v.tolist()
        s_vel_each = reg[("stars", "particle_velocity")].in_units("km/s").v.tolist()

        #Calculating gas' metadata
        g_mass_each = reg[("gas", "cell_mass")].in_units("Msun").v.tolist()
        g_x_each = reg[("gas","x")].in_units("kpc").v.tolist()
        g_y_each = reg[("gas","y")].in_units("kpc").v.tolist()
        g_z_each = reg[("gas","z")].in_units("kpc").v.tolist()
        g_velx_each = reg[("gas","velocity_x")].in_units("km/s").v.tolist()
        g_vely_each = reg[("gas","velocity_y")].in_units("km/s").v.tolist()
        g_velz_each = reg[("gas","velocity_z")].in_units("km/s").v.tolist()
        g_coor_each = []
        g_vel_each = []
        for i in range(len(g_x_each)):
            g_coor_each.append([g_x_each[i],g_y_each[i],g_z_each[i]])
            g_vel_each.append([g_velx_each[i],g_vely_each[i],g_velz_each[i]])
        g_coor_each = np.array(g_coor_each)
        g_vel_each = np.array(g_vel_each)

    if code_name == 'RAMSES':
        ds = yt.load(pfs[int(idx)]) #RAMSES automatically includes the correct conversion factor

        coor = tree['0'][idx]['coor']
        rvir = tree['0'][idx]['Rvir']

        reg = ds.sphere(coor,(rvir,'code_length'))

        com_coor_star = reg.quantities.center_of_mass(use_gas = False, use_particles = True, particle_type='star').to('kpc').v
        com_vel_star = reg.quantities.bulk_velocity(use_gas = False, use_particles = True, particle_type='star').to('km/s').v

        com_coor_bary = reg.quantities.center_of_mass(use_gas = True, use_particles = True, particle_type='star').to('kpc').v
        com_vel_bary = reg.quantities.bulk_velocity(use_gas = True, use_particles = True, particle_type='star').to('km/s').v

        #Calculating stars' metadata
        s_mass_each = reg[("star", "particle_mass")].in_units("Msun").v.tolist()
        s_coor_each = reg[("star", "particle_position")].in_units("kpc").v.tolist()
        s_vel_each = reg[("star", "particle_velocity")].in_units("km/s").v.tolist()

        #Calculating gas' metadata
        g_mass_each = reg[("gas", "cell_mass")].in_units("Msun").v.tolist()
        g_x_each = reg[("gas","x")].in_units("kpc").v.tolist()
        g_y_each = reg[("gas","y")].in_units("kpc").v.tolist()
        g_z_each = reg[("gas","z")].in_units("kpc").v.tolist()
        g_velx_each = reg[("gas","velocity_x")].in_units("km/s").v.tolist()
        g_vely_each = reg[("gas","velocity_y")].in_units("km/s").v.tolist()
        g_velz_each = reg[("gas","velocity_z")].in_units("km/s").v.tolist()
        g_coor_each = []
        g_vel_each = []
        for i in range(len(g_x_each)):
            g_coor_each.append([g_x_each[i],g_y_each[i],g_z_each[i]])
            g_vel_each.append([g_velx_each[i],g_vely_each[i],g_velz_each[i]])
        g_coor_each = np.array(g_coor_each)
        g_vel_each = np.array(g_vel_each)


    #Calculate the angular momentum of only baryonic matters
    if s_mass_each == []: #if there is no star particles
        bary_coor_each = g_coor_each    
        bary_vel_each = g_vel_each
        bary_mass_each = g_mass_each
    else:
        bary_coor_each = np.concatenate((s_coor_each,g_coor_each),axis=0)
        bary_vel_each = np.concatenate((s_vel_each,g_vel_each),axis=0)
        bary_mass_each = np.concatenate((s_mass_each,g_mass_each),axis=0)
    
    bary_relative_coor_each = bary_coor_each - com_coor_bary
    bary_relative_vel_each = bary_vel_each - com_vel_bary
    bary_relative_momentum_each = []
    for i in range(len(bary_mass_each)):
        bary_relative_momentum_each.append(bary_mass_each[i]*bary_relative_vel_each[i])

    bary_relative_momentum_each = np.array(bary_relative_momentum_each)
    #Calculate the angular momentum
    bary_angmoment_each = np.cross(bary_relative_coor_each,bary_relative_momentum_each)

    output_star = {'com_coor_bary':com_coor_bary,'com_coor_star':com_coor_star,'com_vel_bary':com_vel_bary,'com_vel_star':com_vel_star,'mass':s_mass_each,'coor':s_coor_each,'vel':s_vel_each}
    #output_gas = {'com_coor_bary':com_coor_bary,'com_vel_bary':com_vel_bary,'mass':g_mass_each,'coor':g_coor_each,'vel':g_vel_each}
    output_bary = {'com_coor_bary':com_coor_bary,'com_vel_bary':com_vel_bary,'mass':bary_mass_each,'rel_coor':bary_relative_coor_each,'rel_vel':bary_relative_vel_each,'rel_momentum':bary_relative_momentum_each,'angmoment':bary_angmoment_each}

    np.save('metadata/stars_%s.npy' % idx,output_star)
    #np.save('metadata/gas_%s.npy' % idx,output_gas)
    np.save('metadata/bary_%s.npy' % idx,output_bary)
