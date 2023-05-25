import numpy as np
import yt
from yt.data_objects.particle_filters import add_particle_filter

#Create yt filter for Pop 3 stars
def PopIII(pfilter, data):
     filter_stars_pop3 = np.logical_and(data["all", "particle_type"] == 5, data["all", "particle_mass"].to('Msun') > 1) #Adding pop 3 stars
     return filter_stars_pop3  
 
add_particle_filter("PopIII", function=PopIII, filtered_type="all", requires=["particle_type","particle_mass"])

#Create yt filter for stars in general (Pop 2 + Pop 3)
def stars(pfilter, data):
     filter_stars_type_pop2 = data["all", "particle_type"] == 7
     filter_stars_type_pop3 = np.logical_and(data["all", "particle_type"] == 5, data["all", "particle_mass"].to('Msun') > 1) #Adding pop 3 stars
     filter_stars = np.logical_or(filter_stars_type_pop2, filter_stars_type_pop3) 
     return filter_stars  
    
add_particle_filter("stars", function=stars, filtered_type="all", requires=["particle_type","particle_mass"])


def properties_of_halo(halo, sim_data):
    """
    Parameters
    ----------
    halo: list, the halo information from the dictionary
    sim_data: the loaded simulation

    Returns
    -------
    A list containing two sub-lists:
        Sub-list 0: the examined properties of the input halo
        Sub-list 1: the mass and formation time of every star in the halo 
    
    -------
    This function takes the halo information and return the gas and stellar properties
    of a halo

    """
    
    #Load the coordinates and radius of the halo (they are already in code_length unit)
    coor = halo[0]
    radius = halo[2]
    tree_loc = halo[5]
    
    #Select the region of the halo
    reg = sim_data.sphere(coor,(radius,"code_length")) 
    
    #Calculate the gas mass in the region
    g_mass = np.sum(reg[("gas","cell_mass")]).to('Msun') 
    
    #Obtain the type of the particles to get the dark matter particle mass (type_particle == 1)
    type_particle = reg[('all','particle_type')]
    dm_mass = np.sum(reg[("all","particle_mass")][type_particle == 1].to('Msun'))
    
    #Calculate the H2 mass
    h2_mass = np.sum(reg[("gas","H2_mass")]).to('Msun') 
    
    #Calculate the weighted H2_fraction (the weight is the gas mass)
    h2_fraction_each = reg[("gas","H2_fraction")]
    g_mass_each = reg[("gas","cell_mass")].to('Msun')
    h2_fraction = np.average(h2_fraction_each,weights=g_mass_each)
    
    #Calculate the Pop II star mass
    pop2_mass = np.sum(reg[("all","particle_mass")][type_particle == 7].to('Msun'))
    
    #Create the star-type filter for Pop III stars
    sim_data.add_particle_filter("PopIII")
    
    #Calculate the Pop III star mass
    pop3_mass = np.sum(reg["PopIII", "particle_mass"].in_units("Msun"))
    
    #Create the star-type filter to make it easier to extract the creation time for the SFR calculation
    sim_data.add_particle_filter("stars")
    
    #Get the mass and the formation time for each star particle in the halo
    s_mass_each = reg["stars", "particle_mass"].in_units("Msun")
    formation_time = reg["stars", "creation_time"].in_units("Gyr")
    
    #Averaging the SFR 5 million years before the time of the snapshot
    sf_timescale = 0.005*sim_data.units.Gyr
    currenttime = sim_data.current_time.in_units('Gyr')
    
    sfr = np.sum(s_mass_each[formation_time > currenttime - sf_timescale])/sf_timescale 
    sfr = sfr.in_units('Msun/yr')
    
    #Calculate the mvir mass from yt
    mvir = dm_mass + pop2_mass + pop3_mass + g_mass
    
    #Calculate the gas mass fraction
    g_mass_fraction = g_mass/mvir
    
    return tree_loc, coor,float(currenttime),float(g_mass_fraction), float(g_mass), float(dm_mass), float(h2_mass), float(h2_fraction), float(pop2_mass), float(pop3_mass), float(sfr), float(mvir)

#----------------------------------------------------------------------------------
shielding_on = 1

#Create the directory path to each snapshot. The order of the output is the same
#as the order in the pfs.dat file
if shielding_on == 1:
    pfs = open('RT-10_10_v_self_shield/pfs.dat')
    snapshot = []
    for l in pfs:
        l_editted = 'RT-10_10_v_self_shield/' + l.strip()
        snapshot.append(l_editted)

if shielding_on == 0:    
    pfs = open('RT-10_10_v_not_self_shielded/pfs.dat')
    snapshot = []
    for l in pfs:
        l_editted = 'RT-10_10_v_not_self_shielded/' + l.strip()
        snapshot.append(l_editted)


#--------------------------------------------------------------------------
#Running parallel to calculate the properties of each redshift
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size
yt.enable_parallelism(communicator=comm)

#Load the halo list organized by redshift
if shielding_on == 1:
    halolist = np.load('halolist_by_redshift_shield.npy',allow_pickle=True).tolist()
if shielding_on == 0:
    halolist = np.load('halolist_by_redshift_nonshield.npy',allow_pickle=True).tolist()

my_storage = {}

#Running parallel, each processor is responsible for all the halos in one snapshot
for sto, i in yt.parallel_objects(list(halolist.keys()), nprocs-1, storage = my_storage):
    #The key in the halo_ns list is the index of the snapshot in the pfs.dat file (0 corresponds to DD0314, etc.)
    redshift_index = int(i)
    #Each processor obtains all the halos in each snapshot
    all_halos_z = halolist[i]
    #Load the simulation
    sim_data = yt.load(snapshot[redshift_index])
    #Create an array to store the general halo information
    result_each_z = []
    #Run through the list of halo in one snapshot.
    for j in range(len(all_halos_z)):
        halo = all_halos_z[j]
        result_each_z.append(list(properties_of_halo(halo, sim_data))) #Sublist-0 stores the halo's general info
    sto.result = {}
    sto.result[0] = i
    sto.result[1] = result_each_z
    
    
if yt.is_root():
    #Save the halo's general info to one npy file
    halo_list_save = {}
    for j, res in sorted(my_storage.items()):
        halo_list_save[res[0]] = res[1]
    if shielding_on == 1:
        output_name = 'properties_all_halos_shield.npy'
    if shielding_on == 0:
        output_name = 'properties_all_halos_nonshield.npy'
    np.save(output_name,halo_list_save)
    
