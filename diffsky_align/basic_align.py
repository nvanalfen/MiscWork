from pathlib import Path
import opencosmo as oc
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from modular_alignments.modular_alignment import align_to_halo, project_alignments_with_NCP, get_position_angle
from modular_alignments.modular_alignment_2d import align_to_axis, tidal_angle
from modular_alignments.vonmises_distribution import VonMisesHalf
from DW_to_VM_map import DimrothWatsonToVonMisesMapper

def mu_map_generic(mu, *args):
    args = [*args, 0]           # Force mu=0 to map to 0
    return np.polyval(args, mu)

def mu_map_generic_free(mu, *args):
    return np.polyval(args, mu)

# Grab the dataset
data_path = Path("/global/cfs/cdirs/hacc/OpenCosmo/LastJourney/synthetic_galaxies/")
files = list(data_path.glob("*.hdf5"))
dataset = oc.open(*files)

mu = 0.9

# Only take a small subset
dataset = dataset.with_redshift_range(0.1, 0.25)
dataset = dataset.take(100_000)
cols = ["top_host_infall_fof_halo_eigS1X", "top_host_infall_fof_halo_eigS1Y", "top_host_infall_fof_halo_eigS1Z",
       "x_host", "y_host", "z_host", "x", "y", "z", "central", "ra", "dec", "redshift"]
data = dataset.select(cols).get_data("numpy")

# Get values
halo_major_axis = np.array( [ data["top_host_infall_fof_halo_eigS1X"], 
                       data["top_host_infall_fof_halo_eigS1Y"],
                       data["top_host_infall_fof_halo_eigS1Z"] ] ).T
halo_coords = np.array( [ data["x_host"],
                         data["y_host"],
                         data["z_host"] ] ).T
coords = np.array( [ data["x"],
                    data["y"],
                    data["z"] ] ).T
ra = np.array(data["ra"])
dec = np.array(data["dec"])
redshift = np.array(data["redshift"])
centrals = np.array(data["central"])

# Project halo axes
halo_major_proj, NCP, west = project_alignments_with_NCP(halo_major_axis, coords)
halo_phi = get_position_angle(halo_major_proj)

# Prepare 3D to 2D mapper
# Read the complex popt parameters for the major axis projection
popt = np.load("major_axis_projection_misalignment_complex_popt.npz", allow_pickle=True)
primary_popt = popt["primary_popt"]
secondary_popt = popt["secondary_popt"]
weight_popt = popt["weight_popt"]

dwvm = DimrothWatsonToVonMisesMapper(primary_vm_params=primary_popt, secondary_vm_params=secondary_popt, weight_params=weight_popt,
                                     primary_vm_mapper=mu_map_generic, secondary_vm_mapper=mu_map_generic, weight_mapper=mu_map_generic_free)

# Align galaxies
maj,_ = align_to_axis(halo_major_proj, mu, as_vector=True, custom_distr=dwvm)
phi_aligned = get_position_angle(maj)
# e1_aligned, e2_aligned = phi_to_e1_e2(phi_aligned, ellipticity)
np.savez("diffsky_basic.npz", ra=ra, dec=dec, halo_phi=halo_phi, aligned_phi=phi_aligned, centrals=centrals, redshift=redshift)

print("Nothing crashed")