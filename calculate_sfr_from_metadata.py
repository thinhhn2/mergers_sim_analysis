import numpy as np  
import yt

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

my_storage = {}

sf_timescale = 0.01 #Gyr
data = np.load('halotree_Thinh_structure_refinedboundary_2nd.npy',allow_pickle=True).tolist()
snapshot_list = list(data['0'].keys())
output = {}

#for snapshot in snapshot_list:
for sto, snapshot in yt.parallel_objects(snapshot_list, nprocs-1, storage = my_storage):
    currenttime = data['0'][snapshot]['time']
    star_metadata = np.load('metadata/branch-0/' + 'stars_%s.npy' % snapshot, allow_pickle=True).tolist()
    mass = np.array(star_metadata['mass']) #may need to change to initial mass in the future
    formation_time = np.array(star_metadata['formation_time'])
    sfr = mass[formation_time > currenttime - sf_timescale].sum()/(sf_timescale*1e9) #convert sf_timescale to yr
    sto.result = {}
    sto.result['sfr'] = sfr
    sto.result['snapshot'] = snapshot

for c, vals in sorted(my_storage.items()):
    output[vals['snapshot']] = vals['sfr']

if yt.is_root():
    np.save('sfr_from_metadata.npy', output)



