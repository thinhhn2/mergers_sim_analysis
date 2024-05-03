#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 20:32:44 2024

@author: thinhnguyen
"""
import numpy as np
import astropy.units as u
from astropy.constants import G
import matplotlib.pyplot as plt
import seaborn as sns

Gv = G.to('kpc**3/ (Msun*s**2)').value

codetp = 'ENZO'
index_i = '123'
index_f = '142'

def f_rot_cal(index):
    stars_data = np.load('metadata/branch-0/stars_%s.npy' % index,allow_pickle=True).tolist()
    bary_data = np.load('metadata/branch-0/bary_%s.npy' % index,allow_pickle=True).tolist()
    angmoment_data = np.load('metadata/branch-0/angmoment_branch-0.npy',allow_pickle=True).tolist()[index]
    
    angmoment_unitvec = angmoment_data['angmoment_unitvec']
    com_coor_bary = bary_data['com_coor_bary']
    com_vel_bary = bary_data['com_vel_bary']
    
    b_relcoor = bary_data['rel_coor']
    b_mass = bary_data['mass']
    b_r = np.linalg.norm(b_relcoor,axis=1)
    s_coor = stars_data['coor']
    s_mass = stars_data['mass']
    s_vel = stars_data['vel']
    s_relcoor = s_coor - com_coor_bary
    s_r = np.linalg.norm(s_relcoor,axis=1)
    s_relvel = s_vel - com_vel_bary
    s_j = np.cross(s_relcoor, s_relvel)
    s_jz = np.dot(s_j, angmoment_unitvec)
    
    s_jc = np.array([])
    for i in range(len(s_mass)):
        M = b_mass[b_r <  s_r[i]].sum()
        s_jc = np.append(s_jc, u.kpc.to('km')*s_r[i]*np.sqrt(Gv*M/s_r[i]))
        
    f_rot = s_jz/s_jc
    data = {'f_rot':f_rot, 'mass':s_mass}
    return data

#data142 = {'f_rot':list(f_rot), 'mass':s_mass}
#data123 = {'f_rot':f_rot, 'mass':s_mass}
data_i = f_rot_cal(index_i)
data_f = f_rot_cal(index_f)

fig, ax = plt.subplots(figsize=(5*1.2,4.5*1.2))
#sns.displot(data142, x='f_rot', weights='mass', kind = 'kde', gridsize=500, label='142', ax=ax)
#sns.displot(data123, x='f_rot', weights='mass', kind = 'kde', gridsize=500, label='123', ax=ax)
sns.kdeplot(x=data_i['f_rot'], weights=data_i['mass'], gridsize=1000,label='before',ax=ax)
sns.kdeplot(x=data_f['f_rot'], weights=data_f['mass'],gridsize=1000,label='after',ax=ax)
plt.axvline(1,ls='--',color='black')
plt.axvline(-1,ls='--',color='black')
plt.xlim(-5,5)
plt.ylim(0,0.82)
plt.xlabel(r'$f_{rot}=j_z/j_c$',fontsize=14)
plt.ylabel('Density weighted by mass',fontsize=14)
plt.title(codetp,weight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('bulge_fraction_1st_merger.png', dpi=400, bbox_inches='tight')
    
