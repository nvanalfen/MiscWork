from pathlib import Path
import opencosmo as oc
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from modular_alignments.modular_alignment import align_to_halo, project_alignments_with_NCP, get_position_angle, phi_to_e1_e2
from modular_alignments.modular_alignment_2d import align_to_axis, tidal_angle
from modular_alignments.vonmises_distribution import VonMisesHalf
from modular_alignments.alignment_strengths import sigmoid, compound_sigmoid
from DW_to_VM_map import DimrothWatsonToVonMisesMapper
import h5py
import warnings

def zero_vec(vec, size=3):
    if size == 3:
        null = (vec[:,0] == 0) & (vec[:,1] == 0) & (vec[:,2] == 0)
    else:
        null = (vec[:,0] == 0) & (vec[:,1] == 0)
    return sum(null)

def mu_map_generic(mu, *args):
    args = [*args, 0]           # Force mu=0 to map to 0
    return np.polyval(args, mu)

def mu_map_generic_free(mu, *args):
    return np.polyval(args, mu)

# Grab the dataset
main_dir = "/global/cfs/cdirs/hacc/OpenCosmo/LastJourney/synthetic_galaxies/"
sim = "smdpl_dr1_latest"
data_path = Path(main_dir) / sim
# files = list(data_path.glob("*.hdf5"))
files = [f for f in data_path.glob("*.hdf5") if f.stem.startswith("lc_cores")]
dataset = oc.open(*files)
N = len(dataset)
print(data_path)
print(f"N = {N}")

mu = 0.9

# Only take a small subset
# dataset = dataset.with_redshift_range(0.1, 0.25)
selection = dataset.take(N, at="start")
cols = ["top_host_infall_fof_halo_eigS1X", "top_host_infall_fof_halo_eigS1Y", "top_host_infall_fof_halo_eigS1Z",
        "top_host_infall_fof_halo_eigS3X", "top_host_infall_fof_halo_eigS3Y", "top_host_infall_fof_halo_eigS3Z",
       "x_host", "y_host", "z_host", "x", "y", "z", "central", "ra", "dec", "redshift", "logsm_obs",
       "alpha_bulge", "beta_bulge", "alpha_disk", "beta_disk", "lsst_g", "lsst_r"]
data = selection.select(cols).get_data("numpy")

# Get values
halo_major_axis = np.array( [ data["top_host_infall_fof_halo_eigS1X"], 
                       data["top_host_infall_fof_halo_eigS1Y"],
                       data["top_host_infall_fof_halo_eigS1Z"] ] ).T
halo_minor_axis = np.array( [ data["top_host_infall_fof_halo_eigS3X"], 
                       data["top_host_infall_fof_halo_eigS3Y"],
                       data["top_host_infall_fof_halo_eigS3Z"] ] ).T
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
centrals = centrals == 1
logsm_obs = np.array(data["logsm_obs"])
alpha_bulge = np.array(data["alpha_bulge"])
beta_bulge = np.array(data["beta_bulge"])
alpha_disk = np.array(data["alpha_disk"])
beta_disk = np.array(data["beta_disk"])
color_g_min_r = np.array(data["lsst_g"]) - np.array(data["lsst_r"])
# ellipticity = 1 - beta/alpha

mu_cen = 0.7
mu_sat = 0.7

const_mu = False

redshift_params = {"x0":2.0, "k":-1.0, "y_low":0.6, "y_high":1.0}
log_mass_params =  {"x0":6.0, "k":0.5, "y_low":0.}
color_params = {"x0":2.0, "k":-1.0, "y_low":0.2, "y_high":0.8}
if not const_mu:
    mu_cen = compound_sigmoid([color_g_min_r], [color_params])[centrals]
    mu_sat = compound_sigmoid([color_g_min_r], [color_params])[~centrals]

# Project halo axes
halo_major_proj, NCP, west = project_alignments_with_NCP(halo_major_axis[centrals], coords[centrals])
halo_phi = get_position_angle(halo_major_proj)

# Project radial vectors
radial_vec = halo_coords - coords
radial_proj, _, _ = project_alignments_with_NCP(radial_vec[~centrals], coords[~centrals])
radial_phi = get_position_angle(radial_proj)

# Prepare 3D to 2D mapper
# Read the complex popt parameters for the major axis projection
popt = np.load("major_axis_projection_misalignment_complex_popt.npz", allow_pickle=True)
primary_popt = popt["primary_popt"]
secondary_popt = popt["secondary_popt"]
weight_popt = popt["weight_popt"]

dwvm = DimrothWatsonToVonMisesMapper(primary_vm_params=primary_popt, secondary_vm_params=secondary_popt, weight_params=weight_popt,
                                     primary_vm_mapper=mu_map_generic, secondary_vm_mapper=mu_map_generic, weight_mapper=mu_map_generic_free)

# Align central galaxies
maj,_ = align_to_axis(halo_major_proj, mu_cen, as_vector=True, custom_distr=dwvm)
phi_aligned_central = get_position_angle(maj)

# Align radial galaxies
maj, _ = align_to_axis(radial_proj, mu_sat, as_vector=True, custom_distr=dwvm)
phi_aligned_satellite = get_position_angle(maj)

# For getting the diffsky alpha,beta coords
# Alpha
e_alpha_x_disk = np.zeros(len(alpha_disk))
e_alpha_y_disk = np.zeros(len(alpha_disk))
e_alpha_x_bulge = np.zeros(len(alpha_bulge))
e_alpha_y_bulge = np.zeros(len(alpha_bulge))

# Beta
e_beta_x_disk = np.zeros(len(alpha_disk))
e_beta_y_disk = np.zeros(len(alpha_disk))
e_beta_x_bulge = np.zeros(len(alpha_bulge))
e_beta_y_bulge = np.zeros(len(alpha_bulge))

# psi -> position angle is from +y (vertical), psi is from +x
psi_disk = np.zeros(len(alpha_disk))
psi_bulge = np.zeros(len(alpha_bulge))

# Use the position angle to get the right components
# Alpha
e_alpha_x_disk[centrals] = alpha_disk[centrals] * np.sin(phi_aligned_central)    # position angle is from vertical, sin for x, cos for y
e_alpha_y_disk[centrals] = alpha_disk[centrals] * np.cos(phi_aligned_central)
e_alpha_x_disk[~centrals] = alpha_disk[~centrals] * np.sin(phi_aligned_satellite)
e_alpha_y_disk[~centrals] = alpha_disk[~centrals] * np.cos(phi_aligned_satellite)
e_alpha_x_bulge[centrals] = alpha_bulge[centrals] * np.sin(phi_aligned_central)    # position angle is from vertical, sin for x, cos for y
e_alpha_y_bulge[centrals] = alpha_bulge[centrals] * np.cos(phi_aligned_central)
e_alpha_x_bulge[~centrals] = alpha_bulge[~centrals] * np.sin(phi_aligned_satellite)
e_alpha_y_bulge[~centrals] = alpha_bulge[~centrals] * np.cos(phi_aligned_satellite)

# For beta, add pi/2 to phi and use that angle, make sure to shift if it goes beyond pi
beta_phi = np.zeros(len(alpha_disk))
beta_phi[centrals] = phi_aligned_central
beta_phi[~centrals] = phi_aligned_satellite
beta_phi += np.pi/2
mask = beta_phi >= np.pi
beta_phi[mask] -= np.pi

# Beta
# Since I've put them all in the same array, no need to mask centrals
e_beta_x_disk = beta_disk * np.sin(beta_phi)
e_beta_y_disk = beta_disk * np.cos(beta_phi)
e_beta_x_bulge = beta_bulge * np.sin(beta_phi)
e_beta_y_bulge = beta_bulge * np.cos(beta_phi)

# Psi
# Lucky for us, angle from x is simply position angle (from y) + pi/2
# this means beta_phi as already what we want (and we have already shifted!)
psi_disk = np.array(beta_phi)
psi_bulge = np.array(beta_phi)

# For getting e1,e2. Not needed with the current diffsky format, kept here in case
# e1_aligned_cen, e2_aligned_cen = phi_to_e1_e2(phi_aligned_central, ellipticity[centrals])
# e1_aligned_sat, e2_aligned_sat = phi_to_e1_e2(phi_aligned_satellite, ellipticity[~centrals])

# TODO: Save in a way best for Diffsky use. h5 is probably best, I'll ask Andrew
with h5py.File("diffsky_IA.h5", "a") as f:
    f.attrs["main_dir"] = main_dir
    if not sim in f:
        f.create_group(sim)

    group = f[sim]

    if const_mu:
        if not "const_mu" in group:
            group.create_group("const_mu")
        group = group["const_mu"]

        group.attrs["mu_cen"] = mu_cen
        group.attrs["mu_sat"] = mu_sat
    else:
        if not "sigmoid_mu" in group:
            group.create_group("sigmoid_mu")
        group = group["sigmoid_mu"]

        for param in color_params:
            group.attrs[param] = color_params[param]

        group.create_dataset("mu_cen", shape=mu_cen.shape, dtype=mu_cen.dtype, data=mu_cen)
        group.create_dataset("mu_sat", shape=mu_sat.shape, dtype=mu_sat.dtype, data=mu_sat)

    group.create_dataset("e_alpha_x_disk", shape=e_alpha_x_disk.shape, dtype=e_alpha_x_disk.dtype, data=e_alpha_x_disk)
    group.create_dataset("e_alpha_y_disk", shape=e_alpha_y_disk.shape, dtype=e_alpha_y_disk.dtype, data=e_alpha_y_disk)
    group.create_dataset("e_alpha_x_bulge", shape=e_alpha_x_bulge.shape, dtype=e_alpha_x_bulge.dtype, data=e_alpha_x_bulge)
    group.create_dataset("e_alpha_y_bulge", shape=e_alpha_y_bulge.shape, dtype=e_alpha_y_bulge.dtype, data=e_alpha_y_bulge)
    group.create_dataset("e_beta_x_disk", shape=e_beta_x_disk.shape, dtype=e_beta_x_disk.dtype, data=e_beta_x_disk)
    group.create_dataset("e_beta_y_disk", shape=e_beta_y_disk.shape, dtype=e_beta_y_disk.dtype, data=e_beta_y_disk)
    group.create_dataset("e_beta_x_bulge", shape=e_beta_x_bulge.shape, dtype=e_beta_x_bulge.dtype, data=e_beta_x_bulge)
    group.create_dataset("e_beta_y_bulge", shape=e_beta_y_bulge.shape, dtype=e_beta_y_bulge.dtype, data=e_beta_y_bulge)
    group.create_dataset("psi_disk", shape=psi_disk.shape, dtype=psi_disk.dtype, data=psi_disk)
    group.create_dataset("psi_bulge", shape=psi_bulge.shape, dtype=psi_bulge.dtype, data=psi_bulge)

print("No crash")