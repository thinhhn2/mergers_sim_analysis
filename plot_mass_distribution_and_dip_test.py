import numpy as np
import yt
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import diptest
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

def Find_virRad(com,pos,mass,ds,oden=200,radmax=1e50):
    r = np.linalg.norm(com-pos,axis=1)
    r,mass = r[r < radmax],mass[r <radmax]
    arg_r = np.argsort(r)
    mass_cum = mass[arg_r].cumsum()
    V = 4/3 * np.pi * (r)**3
    den = mass_cum/(V[arg_r])
    uden = univDen(ds)
    fden = den/uden
    if fden[fden > oden].size != 0:
        radius = r[arg_r][fden > oden].max()
        cden = fden[fden > oden].min()
    elif len(r) > 0:
        radius = r.max()
        cden = fden.min()
    else:
        radius = 0 * u.m
        cden = 0
    return radius, cden

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
    star_name_dict = {'ENZO':'stars','GADGET3':'PartType4','GEAR':'PartType1','AREPO':'PartType4',\
                    'GIZMO':'PartType4','RAMSES':'star','ART':'stars','CHANGA':'Stars'}
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

codetp = 'ENZO'
os.chdir('/scratch/bbvl/tnguyen2/%s' % codetp)
if not os.path.exists('/scratch/bbvl/tnguyen2/%s/radius_2000' % codetp):
    os.mkdir('/scratch/bbvl/tnguyen2/%s/radius_2000' % codetp)

if codetp == 'ENZO':
    tree = np.load('/scratch/bbvl/tnguyen2/ENZO/halotree_Thinh_structure_with_com.npy',allow_pickle=True).tolist()
elif codetp == 'AREPO':
    tree = np.load('/scratch/bbvl/tnguyen2/ENZO/halotree_Thinh_structure_refinedboundary_2nd_manual_add_2nd_merger.npy',allow_pickle=True).tolist()
else:
    tree = np.load('/scratch/bbvl/tnguyen2/%s/halotree_Thinh_structure_refinedboundary_2nd.npy' % codetp, allow_pickle=True).tolist()

pfs = np.loadtxt('/scratch/bbvl/tnguyen2/%s/pfs_manual.dat' % codetp, dtype=str)

branch_idx = '0'

my_storage= {}

for sto, snapshot_idx in yt.parallel_objects(tree[branch_idx].keys(), nprocs-1, storage = my_storage):
    #
    halo = tree[branch_idx][snapshot_idx]
    if codetp == 'GADGET3' or codetp == 'AREPO':
        ds = yt.load(pfs[int(snapshot_idx)], unit_base = {"length": (1.0, "Mpccm/h")})
    else: 
        ds = yt.load(pfs[int(snapshot_idx)])
    #
    halo_com = halo['coor']
    halo_rvir = halo['Rvir']
    reg = ds.sphere(halo_com,(halo_rvir,'code_length'))
    #
    #Finding R2000 using gas + stars particles
    if codetp == 'ENZO':
        def stars(pfilter, data):
             filter_stars = data["all", "particle_type"] == 2
             return filter_stars
        add_particle_filter("stars", function=stars, filtered_type="all", requires=["particle_type"])
        ds.add_particle_filter("stars")
        #ptype = reg['all', 'particle_type'].v
        #pos_star = reg['all','particle_position'].to('m').v[ptype == 2]
        #mass_star = reg['all','particle_mass'].to('kg').v[ptype == 2]
        pos_star = reg["stars","particle_position"].in_units('m').v
        mass_star = reg["stars","particle_mass"].in_units('kg').v
        x_gas = reg['gas','x'].to('m').v
        y_gas = reg['gas','y'].to('m').v
        z_gas = reg['gas','z'].to('m').v
        pos_gas = np.array([x_gas,y_gas,z_gas]).T
        mass_gas = reg['gas','cell_mass'].to('kg').v
        pos = np.vstack((pos_star,pos_gas))
        mass = np.append(mass_star,mass_gas)
    elif codetp == 'GEAR':
        nm = ParticleUnion("NormalMatter",["PartType1","PartType0"])
        ds.add_particle_union(nm)
        pos = reg['NormalMatter','particle_position'].to('m').v
        mass = reg['NormalMatter','particle_mass'].to('kg').v
    elif codetp == 'GADGET3' or codetp == 'AREPO' or codetp == 'GIZMO':
        nm = ParticleUnion("NormalMatter",["PartType4","PartType0"])
        ds.add_particle_union(nm)
        pos = reg['NormalMatter','particle_position'].to('m').v
        mass = reg['NormalMatter','particle_mass'].to('kg').v
    elif codetp == 'ART':
        pos_star = reg["stars","particle_position"].in_units('m').v
        mass_star = reg["stars","particle_mass"].in_units('kg').v
        x_gas = reg['gas','x'].to('m').v
        y_gas = reg['gas','y'].to('m').v
        z_gas = reg['gas','z'].to('m').v
        pos_gas = np.array([x_gas,y_gas,z_gas]).T
        mass_gas = reg['gas','cell_mass'].to('kg').v
        pos = np.vstack((pos_star,pos_gas))
        mass = np.append(mass_star,mass_gas)
    elif codetp == 'RAMSES':
        pos_star = reg["star","particle_position"].in_units('m').v
        mass_star = reg["star","particle_mass"].in_units('kg').v
        x_gas = reg['gas','x'].to('m').v
        y_gas = reg['gas','y'].to('m').v
        z_gas = reg['gas','z'].to('m').v
        pos_gas = np.array([x_gas,y_gas,z_gas]).T
        mass_gas = reg['gas','cell_mass'].to('kg').v
        pos = np.vstack((pos_star,pos_gas))
        mass = np.append(mass_star,mass_gas)
    elif codetp == 'CHANGA':
        nm = ParticleUnion("NormalMatter",["Stars","Gas"])
        ds.add_particle_union(nm)
        pos = reg['NormalMatter','particle_position'].to('m').v
        mass = reg['NormalMatter','particle_mass'].to('kg').v
    #
    if int(snapshot_idx) == 143 and codetp == 'ENZO':
        initial_gal_com = np.array([0.49442,0.51790,0.50258])
    elif int(snapshot_idx) == 144 and codetp == 'ENZO':
        initial_gal_com = np.array([0.49445,0.51805,0.50263])
    elif int(snapshot_idx) == 145 and codetp == 'ENZO':
        initial_gal_com = np.array([0.49439,0.51815,0.50270])
    elif int(snapshot_idx) == 146 and codetp == 'ENZO':
        initial_gal_com = np.array([0.49429,0.51840,0.50277])
    elif int(snapshot_idx) == 147 and codetp == 'ENZO':
        initial_gal_com = np.array([0.49419,0.51855,0.50285])
    elif int(snapshot_idx) == 148 and codetp == 'ENZO':
        initial_gal_com = np.array([0.49405,0.51869,0.50297])
    elif int(snapshot_idx) == 149 and codetp == 'ENZO':
        initial_gal_com = np.array([0.49393,0.51885,0.50312])
    elif int(snapshot_idx) == 150 and codetp == 'ENZO':
        initial_gal_com = np.array([0.49380,0.51900,0.50332])
    elif int(snapshot_idx) == 151 and codetp == 'ENZO':
        initial_gal_com = np.array([0.49372,0.51917,0.50339])
    else:
        initial_gal_com = Find_Com_HighestDensity(ds, reg)
    #
    initial_gal_com_m = (initial_gal_com*ds.units.code_length).in_units('m').v
    initial_gal_com_kpc = (initial_gal_com*ds.units.code_length).in_units('kpc').v
    halo_rvir_m = (halo_rvir*ds.units.code_length).to('m').v.tolist()
    halo_rvir_kpc = (halo_rvir*ds.units.code_length).in_units('kpc').v.tolist()
    gal_com_m, gal_r2000_m = Find_Com_and_virRad(initial_gal_com_m, halo_rvir_m, pos, mass, oden=2000) #R2000 using star and gas
    gal_com = (gal_com_m*ds.units.m).in_units('code_length').v
    gal_r2000_kpc = (gal_r2000_m*ds.units.m).in_units('kpc').v.tolist()
    gal_r2000 = (gal_r2000_m*ds.units.m).in_units('code_length').v.tolist()
    #Export the data
    output = {}
    output['gal_com'] = gal_com
    output['gal_r2000'] = gal_r2000
    np.save('radius_2000/gal_com_r2000_SnapIdx_%s.npy' % snapshot_idx, output)
    #Plotting the surface mass density in 3 axis
    plt.figure(figsize=(16,5))
    #
    rel_pos_star = pos_star - gal_com_m
    rel_pos_star_kpc = (rel_pos_star*ds.units.m).to('kpc').v
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
    ax1.set_xlim( -np.max(abs(rel_x)), np.max(abs(rel_x)))
    ax1.set_ylim( -np.max(abs(rel_y)), np.max(abs(rel_y)))
    #
    ax2 = plt.subplot(1,3,2)
    _ = ax2.hist2d(rel_y, rel_z, bins=800, norm=mpl.colors.LogNorm(), cmap=mpl.cm.viridis)
    circle2 = plt.Circle((0, 0), gal_r2000_kpc, fill=False)
    ax2.add_patch(circle2)
    ax2.set_xlabel('y (kpc)', fontsize=16)
    ax2.set_ylabel('z (kpc)', fontsize=16)
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.set_xlim( -np.max(abs(rel_y)), np.max(abs(rel_y)))
    ax2.set_ylim( -np.max(abs(rel_z)), np.max(abs(rel_z)))
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
    ax3.set_xlim( -np.max(abs(rel_z)), np.max(abs(rel_z)))
    ax3.set_ylim( -np.max(abs(rel_x)), np.max(abs(rel_x)))
    #
    plt.tight_layout()
    plt.savefig("radius_2000/surface_mass_density_%s_%s.png" % (branch_idx, snapshot_idx),dpi=300,bbox_inches='tight')    
    """
    #using yt to plot, but this method is slow
    fig = plt.figure()
    grid = AxesGrid(
        fig,
        (0.075, 0.075, 0.85, 0.85),
        nrows_ncols=(1, 3),
        axes_pad=1.0,
        label_mode="all",
        share_all=True,
        cbar_location="right",
        cbar_mode="edge",
        cbar_size="5%",
        cbar_pad="0%",
    )
    #
    for i in range(3):
        # Load the data and create a single plot
        p = yt.ParticleProjectionPlot(ds, i, [("stars","particle_mass")], center = (gal_com,'code_length'), width = (2*halo_rvir_kpc, 'kpc'),density=True,fontsize=12)
        p.annotate_sphere(gal_com.tolist(), radius=(gal_r2000_kpc, "kpc"), circle_args={"color": "black"})
        p.set_unit(("stars", "particle_mass"), "Msun/kpc**2")
        p.set_colorbar_label(("stars", "particle_mass"),r'$\Sigma_{*}$ $(\mathrm{M}_\odot/\mathrm{kpc}^{2})$')
        if i == 1:
            p.annotate_title('R2000 = %.2f Rvir' % gal_r2000_kpc/halo_rvir_kpc)
        p.plots[('stars', 'particle_mass')].figure = fig
        p.plots[('stars', 'particle_mass')].axes = grid[i].axes
        p.plots[('stars', 'particle_mass')].cax = grid.cbar_axes[i]
        # Finally, this actually redraws the plot.
        p.render()
    #
    plt.savefig("radius_2000/surface_mass_density_%s_%s.png" % (branch_idx, snapshot_idx),dpi=300,bbox_inches='tight')
    """
    """
    #Make the radial mass distribution plot
    r_m = np.linalg.norm(pos - gal_com_m, axis = 1)
    mass_gal = mass[r_m < gal_r2000_m]
    r_gal_m = r_m[r_m < gal_r2000_m]
    #
    mass_gal_Msun = (mass_gal*ds.units.kg).to('Msun').v
    r_gal_kpc = (r_gal_m*ds.units.m).to('kpc').v
    data = {'r':r_gal_kpc, 'mass':mass_gal_Msun}
    #
    plt.figure(figsize=(9,6))
    ax = sns.histplot(data=data, x = 'r', weights='mass', bins=75, kde=True, alpha=0.2, color='tab:blue',kde_kws={'gridsize':500,'bw_adjust':0.4})
    ax.lines[0].set_color('blue')
    #The reason the KDE fit cannot fit the first peak well is because it tries to fit a Gaussian model
    ax.set_xlabel('r (kpc)',fontsize=14)
    ax.set_ylabel(r'mass ($M_\odot$)',fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.yaxis.get_offset_text().set_fontsize(14)
    #ax.set_title('Hartigan\'s dip test p-value: %.3f' % pval, fontsize=15)
    plt.savefig('mass_distribution_plots_2/radial_mass_distribution_%s_%s.png' % (branch_idx, snapshot_idx), dpi=600, bbox_inches='tight')
    #
    """
    #Using scipy.curve_fit to fit the double Gaussian model
    #from scipy.optimize import curve_fit
    #def bimodal_gaussian(x, amp1, mean1, sigma1, amp2, mean2, sigma2):
    #    return (amp1 * np.exp(-(x - mean1)**2 / (2 * sigma1**2)) + amp2 * np.exp(-(x - mean2)**2 / (2 * sigma2**2)))
    
    #y,x = np.histogram(r_gal_kpc,bins=75,weights=mass_gal_Msun)
    #x=(x[1:]+x[:-1])/2
    #initial_guess = [1e6, 0.1, 0.05, 1e6, 0.5, 0.05]
    #params,cov=curve_fit(bimodal_gaussian,x,y,initial_guess)
    #plt.hist(r_gal_kpc, bins=75, weights=mass_gal_Msun)
    #plt.plot(x,bimodal_gaussian(x,*params),color='red',lw=3,label='model')
    #plt.show()





