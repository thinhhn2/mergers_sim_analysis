import numpy as np
import yt
import glob as glob
import sys,os
import time

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

def make_pfs_allsnaps(folder, codetp, base_manual = None, checkredshift = False):
    """
    This function creates the file that contains the directory to all the simulation snapshots (sorted by redshift).


    Parameters
    ----------
    folder : str
        The directory to the folder containing the simulation snapshots.
    codetp : str
        The code type of the simulation (e.g. 'ENZO', 'GADGET3', 'AREPO', 'RAMSES', 'GIZMO', 'GEAR', 'ART', 'CHANGA').
        If the snapshot base is not in the specified format, set codetp to 'manual'
    base_manual : str, optional
        The base of the snapshot file name. The default is None. Only set if code type is 'manual'.
    checkredshift : bool, optional
        If True, the function will load the snapshhots to check the redshifts and sort the snapshots accordingly. 
        If False, the function will use the index of the snapshot file name to sort the snapshots. The default is False.

    Returns
    -------
    None.

    """
    if yt.is_root():
        os.chdir(folder)
        
        #Obtaining the directory to all the snapshot configuration files
        snapshot_files = []

        if codetp == 'ENZO':
            bases = [["DD", "output_"],
                     ["DD", "data"],
                     ["DD", "DD"],
                     ["RD", "RedshiftOutput"],
                     ["RD", "RD"],
                     ["RS", "restart"]]
            for b in bases:
                snapshot_files += glob.glob("%s????/%s????" % (b[0], b[1]))
                
        elif codetp == 'GADGET3':
            bases = ["snapshot_", "snapshot_"]
            snapshot_files += glob.glob("%s???/%s???.0.hdf5" % (bases[0], bases[1]))
            
        elif codetp == 'AREPO':
            bases = "snap_"
            snapshot_files += glob.glob("%s???.hdf5" % bases)
        
        elif codetp == 'GIZMO':
            bases = "snapshot_"
            snapshot_files += glob.glob("%s???.hdf5" % bases)
            
        elif codetp == 'GEAR':
            bases = "snapshot_"
            snapshot_files += glob.glob("%s????.hdf5" % bases)
    
        elif codetp == 'RAMSES':
            bases = ["output_", "info_"]
            snapshot_files += glob.glob("%s?????/%s?????.txt" % (bases[0], bases[1]))
            
        elif codetp == 'ART':
            bases = '10MpcBox_csf512_'
            snapshot_files += glob.glob('%s?????.d' % bases)
        
        elif codetp == 'CHANGA':
            bases = 'ncal-IV.'
            snapshot_files += glob.glob('%s??????' % bases)
        
        elif codetp == 'manual':
            snapshot_files += glob.glob(base_manual)

        if checkredshift == False: #sort by the index of the snapshot file name
            snapshot_files.sort()
        else: #check the redshift of each snapshot
            #Create a list to store the redshifts of all the snapshots
            snapshot_redshifts = np.array([])

            #Loop through each output. The scale factor is the number after the '#a =' string in the
            #output text file
            if codetp == 'ENZO':
                for file_dir in snapshot_files:
                    with open(file_dir, 'r') as file:
                        for line in file:
                            if line.startswith('CosmologyCurrentRedshift'):
                                redshift = float(line.split()[-1])
                                break
                    snapshot_redshifts = np.append(snapshot_redshifts, redshift) 
            else:
                for file_dir in snapshot_files:
                    ds = yt.load(file_dir)
                    redshift = ds.current_redshift
                    snapshot_redshifts = np.append(snapshot_redshifts, redshift) 

            snapshot_files = np.array(snapshot_files)[np.argsort(-snapshot_redshifts)] #sort in descending order

        #Write out a pfs.dat file
        with open('pfs_allsnaps.dat','w') as file:
            for item in snapshot_files:
                file.write("%s\n" % item)
