##### SKYSIM REALIGN SCRIPT ADJUSTED TO WORK WITH EXTERNAL CONFIG ###############################################################

# Imports
import sys
import os
sys.path.insert(0,'/global/cfs/projectdirs/lsst/groups/SRV/gcr-catalogs')
# reader with parallel read

# Get MPI stuff
from mpi4py import MPI
comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

import GCRCatalogs as gcrc
from astropy.table import Table
from astropy.io import ascii
from astropy.io.misc.hdf5 import write_table_hdf5
import numpy as np
import treecorr
import matplotlib.pyplot as plt
import h5py

from data_utils import (load_yaml_config, check_saved_config, list_extra_columns, extract_requested_columns, get_alignment_masks, 
                        inherit_alignment_config, calculate_mus, remap_mu_tidal, remap_mu_3d, align, rotate_shapes, preprocess_config, 
                        gather_columns_and_filters, save_hdf5)

##### MPI FUNCTIONS #################################################################################################################################

# For MPI
def send_to_master(value, kind):
    """
    Parameters
    ----------
    value : ndarray
        rank-local array of values to communicate
    kind : str
        type of variable. Currently implemented options are double, bool 

    Returns
    -------
    recvbuf : ndarray
        array of all rank values for rank 0 
        None value otherwise
    """
    count = len(value)
    tot_num = comm.reduce(count)
    counts = comm.allgather(count)
    if rank==0:
        if kind=='double':
            recvbuf = np.zeros(tot_num)
        elif kind=='bool':
            recvbuf = np.zeros(tot_num)!=0.0
        elif kind=='int':
            recvbuf = np.zeros(tot_num, dtype='int32')
        elif kind=='int64':
            recvbuf = np.zeros(tot_num, dtype='int64')
        elif kind=='float':
            recvbuf = np.zeros(tot_num, dtype='float32')
        else:
            raise NotImplementedError
    else:
        recvbuf = None

    displs = np.array([sum(counts[:p]) for p in range(size)])
    if kind=='double':
        comm.Gatherv([value,MPI.DOUBLE], [recvbuf,counts,displs,MPI.DOUBLE],root=0)
    elif kind=='float':
        comm.Gatherv([value,MPI.FLOAT], [recvbuf,counts,displs,MPI.FLOAT],root=0)
    elif kind=='int':
        comm.Gatherv([value,MPI.INT], [recvbuf,counts,displs,MPI.INT],root=0)
    elif kind=='bool':
        comm.Gatherv([value,MPI.BOOL], [recvbuf,counts,displs,MPI.BOOL],root=0)
    elif kind=='int64':
        comm.Gatherv([value,MPI.INT64_T], [recvbuf, counts, displs, MPI.INT64_T],root=0)
    else:
        raise NotImplementedError

    return recvbuf

def get_kind(val):
    if type(val)==np.float64:
        return 'double'
    elif type(val)==np.float32:
        return 'float'
    elif type(val)==np.bool_ or type(val)==bool:
        return 'bool'
    elif type(val)==np.int64:
        return 'int64'
    elif type(val)==np.int32:
        return 'int'
    
##### OTHER FUNCTIONS ###############################################################################################################################

# From halotools
def alignment_strength(p):
    r"""
    convert alignment strength argument to shape parameter for costheta distribution
    """
    p = np.atleast_1d(p)
    k = np.zeros(len(p))
    p = p*np.pi/2.0
    k = np.tan(p)
    mask = (p == 1.0)
    k[mask] = np.inf
    mask = (p == -1.0)
    k[mask] = -1.0*np.inf
    return -1.0*k

def cartesian_to_celestial(x, y, z, as_degrees=False):
    factor = 180 / np.pi if as_degrees else 1
    pos = np.array([x, y, z]).T
    rho = np.sqrt( np.sum( pos**2, axis=1 ) )
    ra = np.arctan2(pos[:,1], pos[:,0]) * factor
    ra = np.where(ra < 0, ra+360, ra)
    dec = np.arcsin(pos[:,2] / rho) * factor
    return ra, dec

def correlate(ra, dec, e1, e2, min_sep=0.1, max_sep=3, nbins=20, sep_units="degrees", ra_units="degrees", dec_units="degrees"):
    config = {"min_sep":min_sep, "max_sep":max_sep, "nbins":nbins, "sep_units":"degrees"}
    cat = treecorr.Catalog(ra=ra, dec=dec, g1=e1, g2=e2, ra_units="degrees", dec_units="degrees")
    ng = treecorr.NGCorrelation(config)
    ng.process(cat, cat)
    return ng.logr, ng.xi
#####################################################################################################################################################
##### MAIN ##########################################################################################################################################
#####################################################################################################################################################

def main(config):
    if preprocess_config(config):
        return

    #############################################################################################################################################
    ##### GATHER COLUMNS AND LOAD CATALOG #######################################################################################################
    #############################################################################################################################################
        
    table_columns, native_filters, filters = gather_columns_and_filters(config)

    # Load catalog with mpi parallelization
    cat_name = "skysim5000_v1.2"
    cat = gcrc.load_catalog(cat_name,config_overwrite={'mpi_rank': rank, 'mpi_size': size})
    catalog_data = cat.get_quantities( table_columns, filters=filters, native_filters = native_filters )
            
    data_rank={}
    recvbuf={}
    for quantity in table_columns:
        data_rank[quantity] = catalog_data[quantity]
        if rank==0:
            kind = get_kind(data_rank[quantity][0]) # assumes at least one element of data on rank 0
        else:
            kind = ''
        kind = comm.bcast(kind, root=0)
        recvbuf[quantity] = send_to_master(data_rank[quantity],kind)
        # else:
        #     recvbuf[quantity] = data_rank[quantity]

    # This process function is where the main work is done
    if rank == 0:
        process(catalog_data, config)

def process(data, config):
    # redundant check for rank zero
    if rank != 0:
        print(f"Rank {rank} exiting.", flush=True)
        return
    #########################################################################################################################################
    ##### GET E1, E2, AND MASKS #############################################################################################################
    #########################################################################################################################################

    e1 = data["morphology/totalEllipticity1"]
    e2 = data["morphology/totalEllipticity2"]

    primary_mask, secondary_mask = get_alignment_masks(config, data)

    #########################################################################################################################################
    ##### CALCULATE MU ######################################################################################################################
    #########################################################################################################################################

    # Get an array of mu values
    # The length of mu is the same as the length of data
    # All 0 values where both masks are False
    mus = calculate_mus(config, data, primary_mask, secondary_mask)

    # Remap mu values for both primary and secondary
    remap_mu_tidal(mus, primary_mask, config["alignment_strength"].get("remap_tidal", False))
    remap_mu_tidal(mus, secondary_mask, config["alignment_strength_complement"].get("remap_tidal", False))

    # Get the 3D remap models (a return value of None means we assume the given mu value is already appropriate for a simple 2D alignment)
    primary_remap_model = remap_mu_3d(config["alignment_strength"].get("remap_3d", False))
    secondary_remap_model = remap_mu_3d(config["alignment_strength_complement"].get("remap_3d", False))

    #########################################################################################################################################
    ##### SET UP STORAGE ####################################################################################################################
    #########################################################################################################################################

    # If doing a sparse storage, set the mask we want
    mask = np.ones(len(e1), dtype=bool)
    if config["sparse_storage"]:
        mask = primary_mask|secondary_mask

    # Gather the column data
    column_data = {
        "galaxyID" : data["galaxyID"][mask],
        "mu" : mus[mask]
    }

    #########################################################################################################################################
    ##### ALIGN #############################################################################################################################
    #########################################################################################################################################

    # Loop and do multiple iterations
    loops = config.get("iterations", 1)
    for i in range(loops):
        print(f"Loop {i}", flush=True)
        # Align w.r.t. tidal field
        phi_primary, phi_secondary = align(data, mus, primary_mask, secondary_mask, primary_remap_model, secondary_remap_model)
    
        # Rotate shapes
        e1_primary, e2_primary, e1_secondary, e2_secondary = rotate_shapes(data, phi_primary, phi_secondary, primary_mask, secondary_mask)
    
        # Overwrite e1, e2
        e1[primary_mask] = e1_primary
        e2[primary_mask] = e2_primary
        e1[secondary_mask] = e1_secondary
        e2[secondary_mask] = e2_secondary

        tag = ""
        if loops > 1:
            tag = f"/run_{str(i)}"

        # Write the current iteration
        column_data[f"e1{tag}"] = e1[mask]
        column_data[f"e2{tag}"] = e2[mask]

    #########################################################################################################################################
    ##### WRITE FILE ########################################################################################################################
    #########################################################################################################################################
    
    extra_save_columns = extract_requested_columns(config)
    for col in extra_save_columns:
        column_data[col] = data[col][mask]
    # Write the actual file
    overwrite_matching_file = config.get("overwrite_match", False)
    save_hdf5(config, column_data, check_existing=(not overwrite_matching_file) )

def test_run(config):
    if rank != 0:
        print("Test mode. Non-root rank. Finishing.", flush=True)
        return
    print("Test mode. Root rank. Processing.", flush=True)

    config["output_file"] = "test.h5"
    if preprocess_config(config):
        print("File exists. Proceed anyway for test mode")

    table_columns, native_filters, filters = gather_columns_and_filters(config)
    #############################################################################################################################################
    # Set up dummy data (first 10 rows of some skysim read)
    data_dict = {
        "galaxyID" : np.array([10048000243153, 10048000243159, 10048000243164, 10048000243167,
       10048000243174, 10048000243180, 10048000243188, 10048000243195,
       10048000243197, 10048000243198]),
        "mag_true_g" : np.array([23.004826, 22.331701, 24.339354, 25.303   , 25.436428, 22.435316,
       24.76393 , 23.631788, 23.581116, 24.353907], dtype=np.float32),
        "mag_true_r" : np.array([21.382687, 21.884518, 22.761456, 23.79417 , 24.002644, 21.98813 ,
       23.096352, 22.932407, 23.09235 , 22.712307], dtype=np.float32),
        "redshiftHubble" : np.array([0.80099058, 0.80099058, 0.80099058, 0.80099058, 0.80099058,
       0.80099058, 0.80099058, 0.80099058, 0.80099058, 0.80099058]),
        "tidal_s_11" : np.array([1.51219788, 1.44666211, 1.63791055, 1.44231371, 1.00599651,
       1.50644213, 1.53956033, 1.17460474, 1.33472185, 1.64632943]),
        "tidal_s_12" : np.array([0.60851702, 0.61266809, 0.61130985, 0.60388418, 0.72985301,
       0.60552587, 0.64293075, 0.56982937, 0.60578241, 0.61018375]),
        "tidal_s_22" : np.array([1.13943989, 1.05196306, 1.26268351, 1.08247184, 0.54165417,
       1.05228174, 1.10618964, 0.8346766 , 0.93343482, 1.27541581]),
        "morphology/totalEllipticity" : np.array([0.20381795, 0.08824219, 0.32843938, 0.37253517, 0.12894392,
       0.1846683 , 0.18366903, 0.04851675, 0.49389598, 0.04180954], dtype=np.float32),
        "morphology/totalEllipticity1" : np.array([-0.17713988, -0.08819129,  0.25685507,  0.26360357, -0.03368311,
       -0.15881163,  0.04467028,  0.02475647,  0.27476323,  0.03480506], dtype=np.float32),
        "morphology/totalEllipticity2" : np.array([-0.1008128 , -0.00299667,  0.20468977, -0.2632406 , -0.12446679,
       -0.09424038, -0.17815408,  0.0417252 , -0.4104125 , -0.02316561], dtype=np.float32)
    }
    data = Table(data_dict)
    #############################################################################################################################################
    process(data, config)

if __name__ == "__main__":
    #############################################################################################################################################
    ##### READ CONFIG ###########################################################################################################################
    #############################################################################################################################################
    config_f_name = "config.yaml"
    if len(sys.argv) > 1:
        config_f_name = sys.argv[1]
    config = load_yaml_config(config_f_name)
    if config.get("test",False):
        test_run(config)
    else:
        main(config)