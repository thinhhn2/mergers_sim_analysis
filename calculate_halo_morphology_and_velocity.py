import yt
import numpy as np
import matplotlib.pyplot as plt

star_data = np.load('stars_285.npy',allow_pickle=True).tolist()
gas_data = np.load('gas_285.npy',allow_pickle=True).tolist()
bary_data = np.load('bary_285.npy',allow_pickle=True).tolist()

com_coor_bary = star_data['com_coor_bary']
com_vel_bary = star_data['com_vel_bary']
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
bin_dist = np.linspace(0,0.5*rvir,500)
n_star_distance = []
for i in range(1,len(bin_dist)):
    n_star_bin_distance = len(s_distance_to_angmoment[(s_distance_to_angmoment>bin_dist[i-1]) & (s_distance_to_angmoment<=bin_dist[i])])
    n_star_distance.append(n_star_bin_distance)

n_star_distance_cumsum = np.cumsum(n_star_distance)
n_star_distance_cumsum_percent = n_star_distance_cumsum/len(s_mass_each)

#Find the distance that begins to enclose 99% of the stars in the Rvir region (i.e. scale distance)
percent_lim = 0.99
scale_distance = bin_dist[1:][n_star_distance_cumsum_percent > percent_lim][0]

#Now we re-divide the galaxy into cylindrial shells from the Center of mass to distance99
shell_dist = np.linspace(0,scale_distance,100)

#SUBSECTION 2: WITH RESPECT TO THE HEIGHT OF THE DISK
#WE WANT TO DETERMINE THE SCALE HEIGHT AS A FUNCITON OF DISTANCE
def find_scale_height(distance_start, distance_end):
    """
    This function calculates the scale height of each distance from the rotational axis (i.e. 
    of each cylindrical shell) 
    """
    s_height_distance = s_height[(s_distance_to_angmoment>distance_start) & (s_distance_to_angmoment<=distance_end)]
    bin_height = np.linspace(0,0.5*rvir,500)
    
    n_star_height = []
    for i in range(1,len(bin_height)):
        n_star_bin_height = len(s_height_distance[(s_height_distance>bin_height[i-1]) & (s_height_distance<=bin_height[i])])
        n_star_height.append(n_star_bin_height)
    
    n_star_height_cumsum = np.cumsum(n_star_height)
    n_star_height_cumsum_percent = n_star_height_cumsum/n_star_height_cumsum[-1]
    
    scale_height = bin_height[1:][n_star_height_cumsum_percent > percent_lim][0]
    
    return scale_height

#If we want to find scale height as a function of distance, this is the code   
#scale_height_list = np.array([]) 
#for i in range(1,len(shell_dist)):
#    scale_height = find_scale_height(shell_dist[i-1], shell_dist[i])
#    scale_height_list = np.append(scale_height_list, scale_height)
    
#If we want to use one scale height for the whole galaxy, then 
scale_height_all = find_scale_height(0, scale_distance)

#Calculate the eccentricity of the galaxy, assuming it is an ellipse
eccentricity = np.sqrt(1 - scale_height_all**2/scale_distance**2)    