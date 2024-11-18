import yt
import numpy as np
from yt.data_objects.particle_filters import add_particle_filter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.colors import LogNorm
import copy
import matplotlib
import astropy.units as u 
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

class Find_Galactic_Center():
    """
    """
    def __init__(self, ds, star_pos = None):
        self.ds = ds
        self.star_pos = star_pos if star_pos is not None else None
    #
    def plot_star_particles(self, center, radius):
        plt.figure(figsize=(22.5,8))
        #
        x = self.star_pos[:,0]
        y = self.star_pos[:,1]
        z = self.star_pos[:,2]
        #
        rel_bool = np.linalg.norm(self.star_pos - center, axis=1) < radius
        rel_x = self.star_pos[:,0][rel_bool]
        rel_y = self.star_pos[:,1][rel_bool]
        rel_z = self.star_pos[:,2][rel_bool]
        #
        ax1 = plt.subplot(1,3,1)
        _ = ax1.hist2d(rel_x, rel_y, bins=800, norm=mpl.colors.LogNorm(), cmap=mpl.cm.viridis)
        ax1.set_xlabel('x (code_length)', fontsize=16)
        ax1.set_ylabel('y (code_length)', fontsize=16)
        ax1.tick_params(axis='x', labelsize=16)
        ax1.tick_params(axis='y', labelsize=16)
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.grid(which='both')
        ax1.set_xlim(center[0] - radius, center[0] + radius)
        ax1.set_ylim(center[1] - radius, center[1] + radius)
        ax1.set_aspect('equal') 
        circle1 = plt.Circle((center[0], center[1]), radius, color='r', fill=False)
        ax1.add_patch(circle1)
        #
        ax2 = plt.subplot(1,3,2)
        _ = ax2.hist2d(rel_y, rel_z, bins=800, norm=mpl.colors.LogNorm(), cmap=mpl.cm.viridis)
        ax2.set_xlabel('y (code_length)', fontsize=16)
        ax2.set_ylabel('z (code_length)', fontsize=16)
        ax2.tick_params(axis='x', labelsize=16)
        ax2.tick_params(axis='y', labelsize=16)
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.grid(which='both')
        ax2.set_xlim(center[1] - radius, center[1] + radius)
        ax2.set_ylim(center[2] - radius, center[2] + radius)
        ax2.set_aspect('equal')
        circle2 = plt.Circle((center[1], center[2]), radius, color='r', fill=False)
        ax2.add_patch(circle2)
        #
        ax3 = plt.subplot(1,3,3)
        _ = ax3.hist2d(rel_z, rel_x, bins=800, norm=mpl.colors.LogNorm(), cmap=mpl.cm.viridis)
        ax3.set_xlabel('z (code_length)', fontsize=16)
        ax3.set_ylabel('x (code_length)', fontsize=16)
        ax3.tick_params(axis='x', labelsize=16)
        ax3.tick_params(axis='y', labelsize=16)
        ax3.xaxis.set_minor_locator(AutoMinorLocator())
        ax3.yaxis.set_minor_locator(AutoMinorLocator())
        ax3.grid(which='both')
        ax3.set_xlim(center[2] - radius, center[2] + radius)
        ax3.set_ylim(center[0] - radius, center[0] + radius)
        ax3.set_aspect('equal')
        circle3 = plt.Circle((center[2], center[0]), radius, color='r', fill=False)
        ax3.add_patch(circle3)
        #plt.locator_params(axis='both', nbins=10) 
        plt.tight_layout()   
    #
    def univDen(self):
        # Hubble constant
        H0 = self.ds.hubble_constant * 100 * u.km/u.s/u.Mpc
        H = H0**2 * (self.ds.omega_matter*(1 + self.ds.current_redshift)**3 + self.ds.omega_lambda)  # Technically H^2
        G = 6.67e-11 * u.m**3/u.s**2/u.kg
        # Density of the universe
        den = (3*H/(8*np.pi*G)).to("kg/m**3") / u.kg * u.m**3
        return den.value
        