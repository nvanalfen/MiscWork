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
import numpy as np

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

##### MAIN ##########################################################################################################################################

##### ALIGNMENT VARIABLES ###########################################################################################################################
mu = 0.8

##### LOAD CATALOG ##################################################################################################################################
# Load catalog with mpi parallelization
cat_name = "skysim5000_v1.2"
cat = gcrc.load_catalog(cat_name,config_overwrite={'mpi_rank': rank, 'mpi_size': size})

# Read catalog using filters
# Filter thresholds
REDSHIFT_BLOCK = 0
REDSHIFT_MIN = 0.0
REDSHIFT_MAX = 1.0
DEC_MIN = -36.61
DEC_MAX = 0.0
RA_MIN = 0.0
RA_MAX = 20.0
MAG_R_UPPER = 24.8
MASS_THRESH = 1.846e13
DOWNSAMPLE_FRAC = 0.05                # Downsample if desired [0,1]
print(f"Redshift {REDSHIFT_MIN}-{REDSHIFT_MAX}", flush=True)

# table_columns = [ "redshift", "redshiftHubble", "redshift_true", "ra", "ra_true", "is_central",
#                     "dec", "dec_true", "shear_1", "shear_2", "shear_2_treecorr", "shear1", "shear2",
#                     "tidal_s_11", "tidal_s_12", "tidal_s_22", "mag_true_r", "mag_true_r_sdss", "mag_true_r_lsst" ]
table_columns = [ "baseDC2/target_halo_mass", "redshift", "redshiftHubble", "redshift_true", "ra_true", "is_central",
                    "dec_true", "shear_1", "shear_2", "shear_2_treecorr", "shear1", "shear2",
                    "tidal_s_11", "tidal_s_12", "tidal_s_22", "mag_true_r", "mag_true_r_sdss", "mag_true_r_lsst",
                    "x", "y", "z",
                    "baseDC2/target_halo_x", "baseDC2/target_halo_y", "baseDC2/target_halo_z",
                    "baseDC2/target_halo_axis_A_x", "baseDC2/target_halo_axis_A_y", "baseDC2/target_halo_axis_A_z",
                    "morphology/totalEllipticity", "morphology/totalEllipticity1", "morphology/totalEllipticity2",
                    "galaxyID"]

native_filters = [f'redshift_block_lower <= {REDSHIFT_BLOCK}']
filters = [f"redshiftHubble >= {REDSHIFT_MIN}",f"redshiftHubble < {REDSHIFT_MAX}",
          f"dec_true > {DEC_MIN}", f"dec_true < {DEC_MAX}",
          f"ra_true > {RA_MIN}", f"ra_true < {RA_MAX}",
          #f"mag_true_r_lsst < {MAG_R_UPPER}",
          #f"baseDC2/target_halo_mass > {MASS_THRESH}",
           ]
catalog_data = cat.get_quantities( table_columns, filters=filters, native_filters = native_filters )
        
##### Mask or make final cuts here ########################################
pass

##### Align galaxies in 3D ################################################
# For now, just align all w.r.t. target halo axis
# Later we may split cen and sat for halo and radial alignment
halo_axisA_x = catalog_data["baseDC2/target_halo_axis_A_x"]
halo_axisA_y = catalog_data["baseDC2/target_halo_axis_A_y"]
halo_axisA_z = catalog_data["baseDC2/target_halo_axis_A_z"]
halo_major_axis = np.vstack([halo_axisA_x,halo_axisA_y,halo_axisA_z]).T
galaxy_major_axis, _, _ = align_to_axis_3d(halo_major_axis, alignment_strength(mu))

##### Project to 2D #######################################################
# Just major axis? For now, sure
x = catalog_data["x"]
y = catalog_data["y"]
z = catalog_data["z"]
coords = np.vstack([x,y,z]).T
projected_major, _, _ = project_alignments_with_NCP(galaxy_major_axis, coords)

##### PREPARE MAPPER FOR NEXT STEP ########################################
# These are the parameters I have found from iterative fitting
primary_vm_params = [-0.14538225, 0.22031084, 1., -0.19318874, 0.11735061]
secondary_vm_params = [0.0562699, 0.28310348]
weight_params = [0.42287562, 0.33291063, 0.1770562]
primary_vm_mapper = mu_map_generic
secondary_vm_mapper = mu_map_generic
weight_mapper = mu_map_generic_free
dwvm = DimrothWatsonToVonMisesMapper(primary_vm_params=primary_vm_params, secondary_vm_params=secondary_vm_params, weight_params=weight_params,
                primary_vm_mapper=primary_vm_mapper, secondary_vm_mapper=secondary_vm_mapper, weight_mapper=weight_mapper)

##### Overwrite "bad" orientations by aligning in 2D with tidal fields ####
# Once we have differentiated central and satellites, we should only need to 
# fix centrals (as the positions are more reliable even if the orientation is random, and
# satellites are aligned radially using positions to determine the vector).
rewrite_mask = catalog_data["baseDC2/target_halo_mass"] < MASS_THRESH
sxx = catalog_data["tidal_s_11"][rewrite_mask]
sxy = catalog_data["tidal_s_12"][rewrite_mask]
syy = catalog_data["tidal_s_22"][rewrite_mask]
redshift = catalog_data["redshiftHubble"][rewrite_mask]
tidal_phi = tidal_angle(sxx, sxy, syy, redshift)
tidal_vec = np.array([ np.sin(tidal_phi), np.cos(tidal_phi) ]).T
tidal_corrected_major, _ = align_to_axis_2d(tidal_vec, mu, as_vector=True, custom_distr=dwvm)
projected_major[rewrite_mask] = tidal_corrected_major

# Realign shapes
ellipticity = catalog_data["morphology/totalEllipticity"]
phi = get_position_angle(projected_major)
e1, e2 = phi_to_e1_e2(phi, ellipticity)

# Return subset table
# Include just the rotated shapes? as well as identifying info (halo/galaxy IDs?)
# subset_table = Table({"galaxyID":data["galaxyID"], "e1":e1, "e2":e2})
# f_name = f"ellipticity_add_on_mu={mu}_z={REDSHIFT_MIN}_{REDSHIFT_MAX}.dat"
# ascii.write(data, os.path.join( "/pscratch", "sd", "v", "vanalfen", f_name ), overwrite=True)

data_rank={}
recvbuf={}

# Send the alignments back to the master rank
data_rank["galaxyID"] = catalog_data["galaxyID"]
data_rank["e1"] = e1
data_rank["e2"] = e2
for quantity in data_rank:
    if rank==0:
        kind = get_kind(data_rank[quantity][0]) # assumes at least one element of data on rank 0
    else:
        kind = ''
    kind = comm.bcast(kind, root=0)
    recvbuf[quantity] = send_to_master(data_rank[quantity],kind)

if rank==0:
    data = Table(recvbuf)
    print("Table Length:\t",len(data), flush=True)