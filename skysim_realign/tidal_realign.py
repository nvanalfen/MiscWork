# Imports
import sys
import os
sys.path.insert(0,'/global/cfs/projectdirs/lsst/groups/SRV/gcr-catalogs')
# reader with parallel read

from DW_to_VM_map import DimrothWatsonToVonMisesMapper

from modular_alignments.modular_alignment import project_alignments_with_NCP, get_position_angle, phi_to_e1_e2
from modular_alignments.modular_alignment_2d import tidal_angle
from modular_alignments.modular_alignment import align_to_axis as align_to_axis_3d
from modular_alignments.modular_alignment_2d import align_to_axis as align_to_axis_2d

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
def mu_map_generic(mu, *args):
    args = [*args, 0]           # Force mu=0 to map to 0
    return np.polyval(args, mu)

def mu_map_generic_free(mu, *args):
    return np.polyval(args, mu)

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

##### MAIN ##########################################################################################################################################

##### ALIGNMENT VARIABLES ###########################################################################################################################
mu = 0.8
mu = 0.67*mu        # Account for the fact that I'm aligning directly in 2D

save_plot = False

##### LOAD CATALOG ##################################################################################################################################
# Load catalog with mpi parallelization
cat_name = "skysim5000_v1.2"
cat = gcrc.load_catalog(cat_name,config_overwrite={'mpi_rank': rank, 'mpi_size': size})

# Read catalog using filters
# Filter thresholds
REDSHIFT_BLOCK = 0
REDSHIFT_MIN = -1
REDSHIFT_MAX = 0.5
# DEC_MIN = -36.61
# DEC_MAX = 0.0
# RA_MIN = 0.0
# RA_MAX = 20.0
# MAG_R_UPPER = 24.8
MASS_THRESH = 1.846e13
DOWNSAMPLE_FRAC = 1.0                # Downsample if desired [0,1]
print(f"Redshift {REDSHIFT_MIN}-{REDSHIFT_MAX}", flush=True)

# Just the minimum relevant columns to align to tidal fields in 2D
table_columns = [ "redshiftHubble", "ra_true", "dec_true", "tidal_s_11", "tidal_s_12", "tidal_s_22",
                    "morphology/totalEllipticity", "galaxyID",
                     "x", "y", "z"
                ]

native_filters = [f'redshift_block_lower <= {REDSHIFT_BLOCK}']
# Only filter on redshift for memory sake. I want the full catalog
filters = [f"redshiftHubble >= {REDSHIFT_MIN}",f"redshiftHubble < {REDSHIFT_MAX}",
          # f"dec_true > {DEC_MIN}", f"dec_true < {DEC_MAX}",
          # f"ra_true > {RA_MIN}", f"ra_true < {RA_MAX}",
          #f"mag_true_r_lsst < {MAG_R_UPPER}",
          #f"baseDC2/target_halo_mass > {MASS_THRESH}",
           ]
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
            
if rank==0:
    data = Table(recvbuf)
    
    # Downsample
    #data = data[ data["baseDC2/target_halo_mass"] >= MASS_THRESH ]
    #print(len(data), flush=True)
    # print("Pre-Downsampling:\t",len(data), flush=True)
    # inds = np.random.rand(len(data)) < DOWNSAMPLE_FRAC
    # data = data[inds]
    # print("Post-Downsampling:\t",len(data), flush=True)

    print("Length: ", len(data), flush=True)
    
    
    # if len(data) > 15_000_000:
    #     print("Over 15M galaxies. Don't save.", flush=True)
    # else:
    #     # write out the data to scratch
    #     f_name = f"skysim_selection_z={REDSHIFT_MIN}_{REDSHIFT_MAX}.dat"
    #     ascii.write(data, os.path.join( "/pscratch", "sd", "v", "vanalfen", f_name ), overwrite=True)
    
    ##### Mask or make final cuts here ########################################
    pass
    
    ##### PREPARE MAPPER FOR ALIGNMENT ########################################
    # These are the parameters I have found from iterative fitting
    primary_vm_params = [-0.20252838, 0.10862449, 1., -0.08426155, 0.11907565]
    secondary_vm_params = [0.020614, 0.05596696]
    weight_params = [0.08546417, 0.06449063, 0.08947536]
    primary_vm_mapper = mu_map_generic
    secondary_vm_mapper = mu_map_generic
    weight_mapper = mu_map_generic_free
    dwvm = DimrothWatsonToVonMisesMapper(primary_vm_params=primary_vm_params, secondary_vm_params=secondary_vm_params, weight_params=weight_params,
                    primary_vm_mapper=primary_vm_mapper, secondary_vm_mapper=secondary_vm_mapper, weight_mapper=weight_mapper)

    ##### ALIGN IN 2D WRT TIDAL FIELDS ########################################
    # Once we have differentiated central and satellites, we should only need to 
    # fix centrals (as the positions are more reliable even if the orientation is random, and
    # satellites are aligned radially using positions to determine the vector).
    sxx = data["tidal_s_11"]
    sxy = data["tidal_s_12"]
    syy = data["tidal_s_22"]
    redshift = data["redshiftHubble"]
    tidal_phi = tidal_angle(sxx, syy, sxy, redshift)
    aligned_phi = align_to_axis_2d(tidal_phi, mu, as_vector=False, custom_distr=dwvm)

    # Realign shapes
    ellipticity = data["morphology/totalEllipticity"]
    e1, e2 = phi_to_e1_e2(aligned_phi, ellipticity)

    # Return subset table
    # Include just the rotated shapes as well as identifying info (galaxyID)
    subset_table = Table({"galaxyID":data["galaxyID"], "e1":e1, "e2":e2})
    if REDSHIFT_MIN < 0:
        REDSHIFT_MIN = 0
    # Save as hdf5 file for speed and space
    f_name = f"ellipticity_add_on_mu={mu}_z={REDSHIFT_MIN}_{REDSHIFT_MAX}.h5"
    write_table_hdf5(data, os.path.join( "/pscratch", "sd", "v", "vanalfen", "skysim_realign", f_name ), overwrite=True)

    if save_plot:
        ra, dec = cartesian_to_celestial(data["x"], data["y"], data["z"], as_degrees=True)
        print("Correlating", flush=True)
        logr, xi = correlate(ra, dec, e1, e2)
        np.savez(os.path.join( "/pscratch", "sd", "v", "vanalfen", "skysim_realign", f"corr_mu={mu}_z={REDSHIFT_MIN}_{REDSHIFT_MAX}.npz" ),
                logr=logr, xi=xi)
        plt.figure(figsize=(12, 8))
        plt.plot(np.exp(logr), xi, ":", color="black")
        plt.xscale("log")
        plt.xlabel(r"$\theta$ (deg)")
        plt.ylabel(fr"$\gamma_t$")
        precision = 3
        plt.title(fr"Redshift {np.round(REDSHIFT_MIN, precision)} - {np.round(REDSHIFT_MAX, precision)} : $\mu$ = {mu}")
        plt.savefig(os.path.join( "/pscratch", "sd", "v", "vanalfen", "skysim_realign", "NG.pdf" ), dpi=300)
