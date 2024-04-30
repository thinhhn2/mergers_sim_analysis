import numpy as np  

sf_timescale = 0.01 #Gyr
data = np.load('halotree_Thinh_structure_refinedboundary_2nd.npy',allow_pickle=True).tolist()
snapshot_list = list(data['0'].keys())
output = {}

for snapshot in snapshot_list:
    currenttime = data['0'][snapshot]['time']
    star_metadata = np.load('metadata/branch-0/' + 'stars_%s.npy' % snapshot, allow_pickle=True).tolist()
    mass = np.array(star_metadata['mass']) #may need to change to initial mass in the future
    formation_time = np.array(star_metadata['formation_time'])
    sfr = mass[formation_time > currenttime - sf_timescale].sum()/(sf_timescale*1e9) #convert sf_timescale to yr
    output[snapshot] = sfr

np.save('sfr_from_metadata.npy', output)



