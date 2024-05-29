import yt
import numpy as np
from yt.data_objects.particle_filters import add_particle_filter
import astropy.units as u 

def univDen(ds):
    # Hubble constant
    H0 = ds.hubble_constant * 100 * u.km/u.s/u.Mpc
    H = H0**2 * (ds.omega_matter*(1 + ds.current_redshift)**3 + ds.omega_lambda)  # Technically H^2
    G = 6.67e-11 * u.m**3/u.s**2/u.kg
    # Density of the universe
    den = (3*H/(8*np.pi*G)).to("kg/m**3") / u.kg * u.m**3
    return den.value

def Find_Com_and_virRad(initial_gal_com, halo_rvir, pos, mass, oden=500):
    #Make sure the unit is in kg and m
    virRad = 0.01*halo_rvir
    com = initial_gal_com
    fden = np.inf
    while fden > oden:
        r = np.linalg.norm(pos - com, axis=1)
        pos_in = pos[r < virRad]
        mass_in = mass[r < virRad]
        den = np.sum(mass_in) / (4/3 * np.pi * virRad**3)
        uden = univDen(ds)
        fden = den/uden
        com = np.average(pos_in, weights=mass_in, axis=0)
        virRad += 0.005*halo_rvir
    return com, virRad

def Find_Com_HighestDensity(ds, reg, deposit_dim = 5):
    halo_com = reg.center.to('code_length').v
    halo_rvir = reg.radius.to('code_length').v
    #Divide the region into grids to quickly find the maximum stellar density coordinate
    reg_grid = ds.arbitrary_grid(halo_com - halo_rvir, halo_com + halo_rvir ,dims=[deposit_dim,deposit_dim,deposit_dim])
    density_grid = reg_grid['deposit',star_name_dict[codetp]+'_density'].to('Msun/kpc**3').v
    density_grid_flat = density_grid.flatten()
    #Find the index of maximum density
    maxdensity_idx = np.unravel_index(np.argmax(density_grid_flat), density_grid.shape)
    #Obtain the coordinates of the maximum density
    x_grid = reg_grid['x'].to('code_length').v
    y_grid = reg_grid['y'].to('code_length').v
    z_grid = reg_grid['z'].to('code_length').v
    center_grid = np.stack((x_grid,y_grid,z_grid),axis=3)
    initial_gal_com = center_grid[maxdensity_idx[0],maxdensity_idx[1],maxdensity_idx[2],:]
    return initial_gal_com


tree = np.load('halotree_Thinh_structure_with_com.npy',allow_pickle=True).tolist()
pfs = np.loadtxt('pfs_manual.dat',dtype='str')
snapshot_idx = 150
codetp = 'ENZO'

ds = yt.load(pfs[snapshot_idx])

def stars(pfilter, data):
        filter_stars = data["all", "particle_type"] == 2
        return filter_stars

add_particle_filter("stars", function=stars, filtered_type="all", requires=["particle_type"])
ds.add_particle_filter("stars")

halo = tree['0'][str(snapshot_idx)]
halo_com = np.array(halo['coor'])
halo_rvir = halo['Rvir']

star_name_dict = {'ENZO':'stars','GADGET3':'PartType4','GEAR':'PartType1','AREPO':'PartType4',\
                'GIZMO':'PartType4','RAMSES':'star','ART':'stars','CHANGA':'Stars'}
         
reg = ds.sphere(halo_com, (halo_rvir,'code_length'))
pos_star = reg["stars","particle_position"].in_units('m').v
mass_star = reg["stars","particle_mass"].in_units('kg').v
x_gas = reg['gas','x'].to('m').v
y_gas = reg['gas','y'].to('m').v
z_gas = reg['gas','z'].to('m').v
pos_gas = np.array([x_gas,y_gas,z_gas]).T
mass_gas = reg['gas','cell_mass'].to('kg').v
pos = np.vstack((pos_star,pos_gas))
mass = np.append(mass_star,mass_gas)

#pos and mass are the positions and masses of gas and stars particles
if snapshot_idx == 143 and codetp == 'ENZO':
    initial_gal_com = np.array([0.49442,0.51790,0.50258])
elif snapshot_idx == 144 and codetp == 'ENZO':
    initial_gal_com = np.array([0.49445,0.51805,0.50263])
elif snapshot_idx == 145 and codetp == 'ENZO':
    initial_gal_com = np.array([0.49439,0.51815,0.50270])
elif snapshot_idx == 146 and codetp == 'ENZO':
    initial_gal_com = np.array([0.49429,0.51840,0.50277])
elif snapshot_idx == 147 and codetp == 'ENZO':
    initial_gal_com = np.array([0.49419,0.51855,0.50285])
elif snapshot_idx == 148 and codetp == 'ENZO':
    initial_gal_com = np.array([0.49405,0.51869,0.50297])
elif snapshot_idx == 149 and codetp == 'ENZO':
    initial_gal_com = np.array([0.49393,0.51885,0.50312])
elif snapshot_idx == 150 and codetp == 'ENZO':
    initial_gal_com = np.array([0.49380,0.51900,0.50332])
elif snapshot_idx == 151 and codetp == 'ENZO':
    initial_gal_com = np.array([0.49372,0.51917,0.50339])
else:
    initial_gal_com = Find_Com_HighestDensity(ds, reg)

initial_gal_com_m = (initial_gal_com*ds.units.code_length).to('m').v
halo_rvir_m = (halo_rvir*ds.units.code_length).to('m').v
gal_com_m, gal_rvir_m = Find_Com_and_virRad(initial_gal_com_m, halo_rvir_m, pos, mass, oden=2000)




