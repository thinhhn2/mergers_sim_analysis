import yt
import numpy as np
import matplotlib.pyplot as plt

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

#-------------------------------------------------------------------------------------------
#DEFINE FUNCTIONS

def find_scale_distace(s_distance_to_angmoment,rvir, percent_lim = 0.99):
    """
    This function calculates the scale distance of the galaxy, which is defined as the distance
    that encloses 99% (default value) of the stars in the Rvir region. 

    Note that this is the distance from the rotational axis, not the distance from the center of mass
    """
    bin_dist = np.linspace(0,0.5*rvir,500)
    n_star_distance = []
    for i in range(1,len(bin_dist)):
        n_star_bin_distance = len(s_distance_to_angmoment[(s_distance_to_angmoment>bin_dist[i-1]) & (s_distance_to_angmoment<=bin_dist[i])])
        n_star_distance.append(n_star_bin_distance)

    n_star_distance_cumsum = np.cumsum(n_star_distance)
    n_star_distance_cumsum_percent = n_star_distance_cumsum/len(s_distance_to_angmoment)

    #Find the distance that begins to enclose 99% of the stars in the Rvir region (i.e. scale distance)
    scale_distance = bin_dist[1:][n_star_distance_cumsum_percent > percent_lim][0]

    return scale_distance

def find_scale_height(s_height,s_distance_to_angmoment,distance_start, distance_end, rvir, percent_lim=0.99):
    """
    This function calculates the scale height of each distance from the rotational axis (i.e. 
    of each cylindrical shell). 

    If we want to find the scale height for all stars in the galaxy, then distance_start = 0 and distance_end = scale_distance

    Scale height is defined as the height that encloses 99% (default value) of the stars in the cylindrical shell.
    """
    s_height_distance = s_height[(s_distance_to_angmoment>distance_start) & (s_distance_to_angmoment<=distance_end)]
    bin_height = np.linspace(0,0.5*rvir,500)
    
    n_star_height = []
    for i in range(1,len(bin_height)):
        n_star_bin_height = len(s_height_distance[(s_height_distance>bin_height[i-1]) & (s_height_distance<=bin_height[i])])
        n_star_height.append(n_star_bin_height)
    
    n_star_height_cumsum = np.cumsum(n_star_height)

    n_star_height_cumsum_percent = n_star_height_cumsum/len(s_height_distance)
    #n_star_height_cumsum_percent = n_star_height_cumsum/n_star_height_cumsum[-1]
    
    scale_height = bin_height[1:][n_star_height_cumsum_percent > percent_lim][0]
    
    return scale_height

def weighted_std(values, weights_list):
    """
    Return the weighted standard deviation.

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights_list)
    N = len(values)
    # Fast and numerically precise:
    variance = np.sum(weights_list*(values-average)**2)/((N-1)*np.sum(weights_list)/N)
    return np.sqrt(variance)

def find_average_and_error_of_bins(values, masses, s_galaxy_distance_to_angmoment, shell_dist):
    bins_average = []
    bins_error = []

    for i in range(1,len(shell_dist)):
        bins_value_each = np.array(values)[(s_galaxy_distance_to_angmoment>shell_dist[i-1]) & (s_galaxy_distance_to_angmoment<shell_dist[i])]
        bins_mass_each = np.array(masses)[(s_galaxy_distance_to_angmoment>shell_dist[i-1]) & (s_galaxy_distance_to_angmoment<shell_dist[i])]
        if len(bins_value_each) != 0:
            bins_average.append(np.average(bins_value_each,weights=bins_mass_each))
            bins_error.append(weighted_std(bins_value_each,bins_mass_each))
        else:
            bins_average.append(0)
            bins_error.append(0)

    shell_dist_ave = shell_dist[1:] - (shell_dist[1] - shell_dist[0])/2

    return bins_average, bins_error, shell_dist_ave

#-------------------------------------------------------------------------------------------
#LOAD DATA
tree = np.load('halotree_Thinh_structure_with_com.npy',allow_pickle=True).tolist()
pfs = np.loadtxt('pfs_manual.dat',dtype='str')
snapshot_idx = list(tree['0'].keys())
#-------------------------------------------------------------------------------------------

my_storage = {}
for sto, idx in yt.parallel_objects(snapshot_idx, nprocs-1,storage = my_storage):

    star_data = np.load('stars_%s.npy' % idx,allow_pickle=True).tolist()
    gas_data = np.load('gas_%s.npy' % idx,allow_pickle=True).tolist()
    bary_data = np.load('bary_%s.npy' % idx,allow_pickle=True).tolist()

    com_coor_bary = star_data['com_coor_bary']
    com_vel_bary = star_data['com_vel_bary']
    com_coor_star = star_data['com_coor_star']
    com_vel_star = star_data['com_vel_star']
    s_mass_each = star_data['mass']
    s_coor_each = star_data['coor']
    s_vel_each = star_data['vel']
    s_rel_coor_each = s_coor_each - com_coor_bary
    s_rel_vel_each = s_vel_each - com_vel_bary

    g_mass_each = gas_data['mass']
    g_coor_each = gas_data['coor']
    g_vel_each = gas_data['vel']

    bary_rel_coor_each = bary_data['rel_coor']
    bary_rel_vel_each = bary_data['rel_vel']
    bary_rel_momentum_each = bary_data['rel_momentum']
    bary_angmoment_each = bary_data['angmoment']

    bary_angmoment = np.sum(bary_angmoment_each,axis=0)
    bary_angmoment_unitvec = bary_angmoment/np.sum(bary_angmoment**2)**0.5

    #Distance from the particles to the rotational axis
    numerator = np.cross(bary_relative_coor_each,bary_angmoment_unitvec)
    distance_to_angmoment = np.sqrt(np.sum(numerator**2,axis = 1))/np.sqrt(np.sum(bary_angmoment_unitvec**2))

    #Distance between the particles to the disk
    height = np.abs(np.dot(bary_relative_coor_each,bary_angmoment_unitvec))

    #Same for stars
    s_numerator = np.cross(s_relative_coor_each,bary_angmoment_unitvec)
    s_distance_to_angmoment = np.sqrt(np.sum(s_numerator**2,axis = 1))/np.sqrt(np.sum(bary_angmoment_unitvec**2))
    s_height = np.abs(np.dot(s_relative_coor_each,bary_angmoment_unitvec)) #note that the distance can be negative (i.e under the equatorial plane)

    #THIS SECTION DETERMINES WHICH REGION OF THE HALO MOST OF THE GALAXY RESIDE IN 
    #SUBSECTION 1: WITH RESPECT TO THE DISTANCE TO THE ROTATIONAL AXIS
    scale_distance = find_scale_distace(s_distance_to_angmoment,rvir)
    #Now we re-divide the galaxy into cylindrial shells from the Center of mass to distance99
    shell_dist = np.linspace(0,scale_distance,100)

    #SUBSECTION 2: WITH RESPECT TO THE HEIGHT OF THE DISK
    #WE WANT TO DETERMINE THE SCALE HEIGHT AS A FUNCITON OF DISTANCE

    #If we want to find scale height as a function of distance, this is the code   
    #scale_height_list = np.array([]) 
    #for i in range(1,len(shell_dist)):
    #    scale_height = find_scale_height(shell_dist[i-1], shell_dist[i])
    #    scale_height_list = np.append(scale_height_list, scale_height)
        
    #If we want to use one scale height for the whole galaxy, then 
    scale_height_all = find_scale_height(s_height,s_distance_to_angmoment,0, scale_distance, rvir)

    #Calculate the eccentricity of the galaxy, assuming it is an ellipse
    eccentricity = np.sqrt(1 - scale_height_all**2/scale_distance**2)    

    #The velocity dispersion of only stars
    s_dispersion = []

    for i in range(1,len(shell_dist)):
        s_bin_vel = np.array(s_relative_vel_each)[(s_distance_to_angmoment>shell_dist[i-1]) & (s_distance_to_angmoment<shell_dist[i]) & (s_height <= scale_height_all)]
        s_bin_mass = np.array(s_mass_each)[(s_distance_to_angmoment>shell_dist[i-1]) & (s_distance_to_angmoment<shell_dist[i]) & (s_height <= scale_height_all)]
        s_bin_vel_magnitude = np.sum(s_bin_vel**2,axis=1)**0.5
        if len(s_bin_vel) != 0:
            s_bin_dispersion = weighted_std(s_bin_vel_magnitude, s_bin_mass)
        else:
            s_bin_dispersion = 0
        s_dispersion.append(s_bin_dispersion)
        

    s_dispersion = np.nan_to_num(np.array(s_dispersion))
    #Normalize by the virial radius (normalize from 0 to 1)
    shell_dist_norm = shell_dist/rvir

    plt.stairs(s_dispersion,shell_dist_norm)
    plt.ylabel('Velocity dispersion of stars (km/s)',fontsize=14)
    plt.xlabel('r/Rvir',fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('velocity_dispersion_%s.svg' % idx)

    #-------------------------------------------------------------------------------------------
    #THIS SECTION CALCULATES THE ROTATIONAL AXIS COMPONENT OF THE VELOCITY 

    #Create a boolean array to select the stars within the scale radius and scale height
    s_galaxy = (s_distance_to_angmoment>0) & (s_distance_to_angmoment<=scale_distance) & (s_height <= scale_height_all)
    s_galaxy_relative_vel_each = np.array(s_relative_vel_each)[s_galaxy]
    s_galaxy_relative_coor_each = np.array(s_relative_coor_each)[s_galaxy]
    s_galaxy_mass_each = np.array(s_mass_each)[s_galaxy]
    s_galaxy_distance_to_angmoment = np.array(s_distance_to_angmoment)[s_galaxy]
    s_galaxy_L_vel_each = np.dot(s_galaxy_relative_vel_each,bary_angmoment_unitvec)

    from matplotlib.colors import LogNorm
    plt.hist2d(s_galaxy_distance_to_angmoment/rvir,s_galaxy_L_vel_each,bins=[200,200],norm=LogNorm())
    plt.xlabel('r/Rvir',fontsize=14)
    plt.ylabel(r'$v_{L}$ (km/s)',fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('velocity_rotaxis_histogram_%s.svg' % idx)

    s_bin_galaxy_L_vel = []
    s_bin_galaxy_L_vel_error = []

    for i in range(1,len(shell_dist)):
        s_bin_galaxy_L_vel_each = np.array(s_galaxy_L_vel_each)[(s_galaxy_distance_to_angmoment>shell_dist[i-1]) & (s_galaxy_distance_to_angmoment<shell_dist[i])]
        s_bin_galaxy_mass_each = np.array(s_galaxy_mass_each)[(s_galaxy_distance_to_angmoment>shell_dist[i-1]) & (s_galaxy_distance_to_angmoment<shell_dist[i])]
        if len(s_bin_galaxy_disk_vel_each) != 0:
            s_bin_galaxy_L_vel.append(np.average(s_bin_galaxy_L_vel_each,weights=s_bin_galaxy_mass_each))
            s_bin_galaxy_L_vel_error.append(weighted_std(s_bin_galaxy_L_vel_each,s_bin_galaxy_mass_each))
        else:
            s_bin_galaxy_L_vel.append(0)
            s_bin_galaxy_L_vel_error.append(0)


    shell_dist_ave = shell_dist[1:] - (shell_dist[1] - shell_dist[0])/2

    plt.errorbar(shell_dist_ave/rvir,s_bin_galaxy_L_vel,yerr = s_bin_galaxy_L_vel_error)
    plt.xlabel('r/Rvir', fontsize=14)
    plt.ylabel(r'$v_{L}$ (km/s)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('velocity_rotaxis_%s.svg' % idx)
    #--------------------------------------------------------------------------------------------
    #THIS SECTION CALCULATES THE DISK-COMPONENT OF THE VELOCITY (PERPENDICULAR TO THE ROTATIONAL-AXIS COMPONENT)

    s_galaxy_disk_vel_numerator = np.cross(s_galaxy_relative_vel_each,bary_angmoment_unitvec)
    s_galaxy_disk_vel_each = np.sum(s_galaxy_disk_vel_numerator**2,axis=1)**0.5

    s_bin_galaxy_disk_vel = []
    s_bin_galaxy_disk_vel_error = []

    for i in range(1,len(shell_dist)):
        s_bin_galaxy_disk_vel_each = np.array(s_galaxy_disk_vel_each)[(s_galaxy_distance_to_angmoment>shell_dist[i-1]) & (s_galaxy_distance_to_angmoment<shell_dist[i])]
        s_bin_galaxy_mass_each = np.array(s_galaxy_mass_each)[(s_galaxy_distance_to_angmoment>shell_dist[i-1]) & (s_galaxy_distance_to_angmoment<shell_dist[i])]
        if len(s_bin_galaxy_disk_vel_each) != 0:
            s_bin_galaxy_disk_vel.append(np.average(s_bin_galaxy_disk_vel_each,weights=s_bin_galaxy_mass_each))
            s_bin_galaxy_disk_vel_error.append(weighted_std(s_bin_galaxy_disk_vel_each,s_bin_galaxy_mass_each))
        else:
            s_bin_galaxy_disk_vel.append(0)
            s_bin_galaxy_disk_vel_error.append(0)


    shell_dist_ave = shell_dist[1:] - (shell_dist[1] - shell_dist[0])/2

    plt.errorbar(shell_dist_ave/rvir,s_bin_galaxy_disk_vel,yerr = s_bin_galaxy_disk_vel_error)
    plt.xlabel('r/Rvir', fontsize=14)
    plt.ylabel(r'$v_{disk-plane}$ (km/s)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('velocity_disk_%s.svg' % idx)
    #-------------------------------------------------------------------------------------------
    #THIS SECTION CALCULATES THE RATIO BETWEEN THE L-COMPONENT AND THE TOTAL MAGNITUDE OF THE VELOCITY 
    s_galaxy_vel_magnitude_each = np.sum(s_galaxy_relative_vel_each**2,axis=1)**0.5
    s_galaxy_vel_L_ratio_each = np.abs(s_galaxy_L_vel_each/s_galaxy_vel_magnitude_each)

    s_bin_galaxy_vel_L_ratio = []
    s_bin_galaxy_vel_L_ratio_error = []

    shell_dist = np.linspace(0,scale_distance,100)
    for i in range(1,len(shell_dist)):
        s_bin_galaxy_vel_L_ratio_each = np.array(s_galaxy_vel_L_ratio_each)[(s_galaxy_distance_to_angmoment>shell_dist[i-1]) & (s_galaxy_distance_to_angmoment<shell_dist[i])]
        s_bin_galaxy_mass_each = np.array(s_galaxy_mass_each)[(s_galaxy_distance_to_angmoment>shell_dist[i-1]) & (s_galaxy_distance_to_angmoment<shell_dist[i])]
        if len(s_bin_galaxy_vel_L_ratio_each) != 0:
            s_bin_galaxy_vel_L_ratio.append(np.average(s_bin_galaxy_vel_L_ratio_each,weights=s_bin_galaxy_mass_each))
            s_bin_galaxy_vel_L_ratio_error.append(weighted_std(s_bin_galaxy_vel_L_ratio_each,s_bin_galaxy_mass_each))
        else:
            s_bin_galaxy_vel_L_ratio.append(0)
            s_bin_galaxy_vel_L_ratio_error.append(0)


    shell_dist_ave = shell_dist[1:] - (shell_dist[1] - shell_dist[0])/2
    plt.errorbar(shell_dist_ave/rvir,s_bin_galaxy_vel_L_ratio,yerr = s_bin_galaxy_vel_L_ratio_error)
    plt.xlabel('r/Rvir', fontsize=14)
    plt.ylabel(r'$v_{L}/|v|$', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('velocity_rotaxis_ratio_%s.svg' % idx)

    #-------------------------------------------------------------------------------------------
    #THIS SECTION CALCULATES THE RATIO BETWEEN THE TOTAL ANGULAR MOMENTUM VS THE L-COMPONENT ANGULAR MOMENTUM
    s_galaxy_relative_momentum_each = []
    for i in range(len(s_galaxy_mass_each)):
        s_galaxy_relative_momentum_each.append(s_galaxy_mass_each[i]*s_galaxy_relative_vel_each[i])


    s_galaxy_relative_momentum_each = np.array(s_galaxy_relative_momentum_each)
    #Calculate the angular momentum
    s_galaxy_angmoment_each = np.cross(s_galaxy_relative_coor_each,s_galaxy_relative_momentum_each)

    s_galaxy_L_angmoment_each = np.abs(np.dot(s_galaxy_angmoment_each,bary_angmoment_unitvec))
    s_galaxy_angmoment_magnitude_each = np.sqrt(np.sum(s_galaxy_angmoment_each**2,axis=1))
    s_galaxy_angmoment_L_ratio = s_galaxy_L_angmoment_each/s_galaxy_angmoment_magnitude_each

    s_bin_galaxy_angmoment_L_ratio = []
    s_bin_galaxy_angmoment_L_ratio_error = []

    for i in range(1,len(shell_dist)):
        s_bin_galaxy_angmoment_L_ratio_each = np.array(s_galaxy_angmoment_L_ratio)[(s_galaxy_distance_to_angmoment>shell_dist[i-1]) & (s_galaxy_distance_to_angmoment<shell_dist[i])]
        s_bin_galaxy_mass_each = np.array(s_galaxy_mass_each)[(s_galaxy_distance_to_angmoment>shell_dist[i-1]) & (s_galaxy_distance_to_angmoment<shell_dist[i])]
        if len(s_bin_galaxy_angmoment_L_ratio_each) != 0:
            s_bin_galaxy_angmoment_L_ratio.append(np.average(s_bin_galaxy_angmoment_L_ratio_each,weights=s_bin_galaxy_mass_each))
            s_bin_galaxy_angmoment_L_ratio_error.append(weighted_std(s_bin_galaxy_angmoment_L_ratio_each,s_bin_galaxy_mass_each))
        else:
            s_bin_galaxy_angmoment_L_ratio.append(0)
            s_bin_galaxy_angmoment_L_ratio_error.append(0)


    shell_dist_ave = shell_dist[1:] - (shell_dist[1] - shell_dist[0])/2
    plt.errorbar(shell_dist_ave/rvir,s_bin_galaxy_angmoment_L_ratio,yerr = s_bin_galaxy_angmoment_L_ratio_error)
    plt.xlabel('r/Rvir', fontsize=14)
    plt.ylabel(r'|$J_{L}$|/|$J$|', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('angmoment_rotaxis_ratio_%s.svg' % idx)

    #-------------------------------------------------------------------------------------------
    #THIS SECTION CALCULATE THE KAPPA PARAMETER
    #KAPPA PARAMETER < 0.5 -> DISK. KAPPA PARAMETER >= 0.5 -> SPHEROID
    s_galaxy_K_each = 0.5*s_galaxy_mass_each*s_galaxy_vel_magnitude_each**2
    s_galaxy_K = np.sum(s_galaxy_K_each)

    s_galaxy_specific_L_angmoment_each = s_galaxy_L_angmoment_each/s_galaxy_mass_each
    s_galaxy_Krot_each = 0.5*s_galaxy_mass_each*(s_galaxy_specific_L_angmoment_each/s_galaxy_distance_to_angmoment)**2
    s_galaxy_Krot = np.sum(s_galaxy_Krot_each)

    kappa_param = s_galaxy_Krot/s_galaxy_K

    #-------------------------------------------------------------------------------------------
    #THIS SECTION CALCULATES THE DISTANCE BETWEEN THE BARY COM AND THE STAR COM
    dist_barycom_starcom = np.sqrt(np.sum((com_coor_bary - com_coor_star)**2))

    #-------------------------------------------------------------------------------------------
    #THIS SECTION CALCULATES THE BULLOCK SPIN PARAMETER FOR BARYONIC MATTER
    spin_dist = scale_distance #The distance where we evaluate the Bullock spin parameter

    bary_distance_to_com = np.sqrt(np.sum(bary_rel_coor_each**2))
    bary_spin = bary_distance_to_com < spin_dist
    bary_spin_mass_each = np.array(bary_mass_each)[bary_spin]
    bary_spin_mass = np.sum(bary_spin_mass_each)

    bary_spin_angmoment_each = bary_angmoment_each[bary_spin]

    bary_spin_angmoment_magnitude = np.sum(bary_angmoment**2)**0.5

    #Calculate the Bullock Spin Parameter. Note that this is spin*sqrt(G)
    spin_param = bary_angmoment_magnitude/np.sqrt(2*spin_dist*bary_spin_mass**3)

    #Built-in function to calculate the regular spin parameter (from Peebles 1971, not from Bullock 2001)
    ds = yt.load('RD0285/RD0285')
    reg_spin = ds.sphere(com_coor_bary,(scale_distance,'code_length'))
    spin_param_yt = reg_spin.quantities.spin_parameter(use_gas = True, use_particles = True, particle_type='stars')

    output = {}
    output['scale_distance'] = scale_distance
    output['scale_height'] = scale_height_all
    output['eccentricity'] = eccentricity
    output['kapppa_param'] = kappa_param
    output['dist_barycom_starcom'] = dist_barycom_starcom
    output['spin_param'] = spin_param
    output['spin_param_yt'] = spin_param_yt

    np.save('morphology/morphology_%s.npy' % idx,output)







