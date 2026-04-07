from pathlib import Path
import opencosmo as oc
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from modular_alignments.modular_alignment import align_to_halo, project_alignments_with_NCP, get_position_angle, phi_to_e1_e2
from modular_alignments.modular_alignment_2d import align_to_axis, tidal_angle
from modular_alignments.modular_alignment import align_to_axis as align_to_axis_3D
from modular_alignments.vonmises_distribution import VonMisesHalf
from modular_alignments.alignment_strengths import sigmoid, compound_sigmoid
from DW_to_VM_map import DimrothWatsonToVonMisesMapper
import warnings
from shape_draw import get_empirical_percentiles, MixedBetaDistribution
from project_full_ellipses import compute_ellipse2d
from halotools.utils.vector_utilities import normalized_vectors

def mu_map_generic(mu, *args):
    args = [*args, 0]           # Force mu=0 to map to 0
    return np.polyval(args, mu)

def mu_map_generic_free(mu, *args):
    return np.polyval(args, mu)

def get_e_percentiles(axisA, axisC, dataset, popN=100_000):
    """
    Given the sample loaded, use the halo c/a ratio as a proxy for ellipticity.
    Load a much larger population and get their c/a ratios. Find what percentile
    each sample c/a is within the larger population (assuming this will be fairly
    representative of its true percentile). These will be used to draw galaxy shapes
    by assuming a halo with 85th percentile e will correspond to its galaxy being
    85th percentile in its distribution as well.

    Inputs
    ------
    axisA:        array, Nx3 array of the major axis.
    axisC:        array, Nx3 array of the minor axis.
    dataset:      opencosmo dataset. The same cuts used to grab the data being sampled.
                  Use to draw a larger population from which to sample more halo shapes.
    popN:         The size of the larger population to draw. If this is equal to or smaller
                  than the length of the sample, no new points will be drawn.

    Returns:
    --------
    percentiles:  array, length N array of the percentiles into which the c/a shapes fall
                  in the larger distribution.
    """
    assert axisA.shape[1] == 3, "Axis array must be Nx3"
    assert axisC.shape[1] == 3, "Axis array must be Nx3"
    c_to_a = np.linalg.norm(axisC, axis=1) / np.linalg.norm(axisA, axis=1)

    if popN <= len(c_to_a):
        warnings.warn("""sample population equal to or greater than popN. No extra halos will be drawn
        and percentiles will instead be drawn from sample directly. This may lead to unexpected behavior
        or unreliable mappings.""")
        return get_empirical_percentiles(c_to_a, c_to_a)

    # Load just the axes for popN halos
    temp_dataset = dataset.take(popN)
    cols = ["top_host_infall_fof_halo_eigS1X", "top_host_infall_fof_halo_eigS1Y", "top_host_infall_fof_halo_eigS1Z",
        "top_host_infall_fof_halo_eigS3X", "top_host_infall_fof_halo_eigS3Y", "top_host_infall_fof_halo_eigS3Z"]
    data = dataset.select(cols).get_data("numpy")

    # Get the shape distributions from the larger halo population
    axisA_pop = np.array([ data["top_host_infall_fof_halo_eigS1X"], data["top_host_infall_fof_halo_eigS1Y"], data["top_host_infall_fof_halo_eigS1Z"] ]).T
    axisC_pop = np.array([ data["top_host_infall_fof_halo_eigS3X"], data["top_host_infall_fof_halo_eigS3Y"], data["top_host_infall_fof_halo_eigS3Z"] ]).T
    c_to_a_pop = np.linalg.norm(axisC_pop, axis=1) / np.linalg.norm(axisA_pop, axis=1)

    # Get the percentile ellipticity of the given axisC/axisA shapes from the larger population
    return get_empirical_percentiles(c_to_a_pop, c_to_a)

# Grab the dataset
data_path = Path("/global/cfs/cdirs/hacc/OpenCosmo/LastJourney/synthetic_galaxies_1000deg2_unlensed")
# files = list(data_path.glob("*.hdf5"))
files = [f for f in data_path.glob("*.hdf5") if f.stem.startswith("lc_cores")]
dataset = oc.open(*files)

mu = 0.9

# Only take a small subset
dataset = dataset.with_redshift_range(0.1, 0.25)
min_mass = oc.col("logsm_obs") > 10
dataset = dataset.filter(min_mass)
selection = dataset.take(100_000)
cols = ["top_host_infall_fof_halo_eigS1X", "top_host_infall_fof_halo_eigS1Y", "top_host_infall_fof_halo_eigS1Z",
        "top_host_infall_fof_halo_eigS2X", "top_host_infall_fof_halo_eigS2Y", "top_host_infall_fof_halo_eigS2Z",
        "top_host_infall_fof_halo_eigS3X", "top_host_infall_fof_halo_eigS3Y", "top_host_infall_fof_halo_eigS3Z",
       "x_host", "y_host", "z_host", "x", "y", "z", "central", "ra", "dec", "redshift", "logsm_obs"]
data = selection.select(cols).get_data("numpy")

# Get values
halo_major_axis = np.array( [ data["top_host_infall_fof_halo_eigS1X"], 
                       data["top_host_infall_fof_halo_eigS1Y"],
                       data["top_host_infall_fof_halo_eigS1Z"] ] ).T
halo_inter_axis = np.array( [ data["top_host_infall_fof_halo_eigS2X"], 
                       data["top_host_infall_fof_halo_eigS2Y"],
                       data["top_host_infall_fof_halo_eigS2Z"] ] ).T
halo_minor_axis = np.array( [ data["top_host_infall_fof_halo_eigS3X"], 
                       data["top_host_infall_fof_halo_eigS3Y"],
                       data["top_host_infall_fof_halo_eigS3Z"] ] ).T
halo_coords = np.array( [ data["x_host"],
                         data["y_host"],
                         data["z_host"] ] ).T
coords = np.array( [ data["x"],
                    data["y"],
                    data["z"] ] ).T

mask = (np.all(halo_major_axis == 0, axis=1)) | (np.all(halo_inter_axis == 0, axis=1)) | (np.all(halo_minor_axis == 0, axis=1))
halo_major_axis[mask] = np.array([1,0,0])
halo_inter_axis[mask] = np.array([0,0.5,0])
halo_minor_axis[mask] = np.array([0,0,0.25])

print("NANS")
print(np.isnan(halo_major_axis).any(axis=1).sum())
print(np.isnan(halo_inter_axis).any(axis=1).sum())
print(np.isnan(halo_minor_axis).any(axis=1).sum())
print(np.isnan(halo_coords).any(axis=1).sum())
print(np.isnan(coords).any(axis=1).sum())
print("ZEROS")
print(np.all(halo_major_axis == 0, axis=1).sum())
print(np.all(halo_inter_axis == 0, axis=1).sum())
print(np.all(halo_minor_axis == 0, axis=1).sum())
print(np.all(halo_coords == 0, axis=1).sum())
print(np.all(coords == 0, axis=1).sum())

ra = np.array(data["ra"])
dec = np.array(data["dec"])
redshift = np.array(data["redshift"])
centrals = np.array(data["central"])
logsm_obs = np.array(data["logsm_obs"])

redshift_params = {"x0":2.0, "k":-1.0, "y_low":0.6, "y_high":1.0}
log_mass_params =  {"x0":6.0, "k":0.5, "y_low":0.}
mu = compound_sigmoid([redshift, logsm_obs], [redshift_params, log_mass_params])

# plt.figure(figsize=(10,7))
# plt.scatter(redshift, sigmoid(redshift, **redshift_params))
# plt.xlabel("redshift")
# plt.ylabel("y_high")
# plt.savefig("temp_redshift_plot.pdf", dpi=300)

# plt.figure(figsize=(10,7))
# plt.scatter(logsm_obs, sigmoid(logsm_obs, **log_mass_params))
# plt.xlabel("logsm_obs")
# plt.ylabel("y_high")
# plt.savefig("temp_log_sm_plot.pdf", dpi=300)

# plt.figure(figsize=(10,7))
# plt.scatter(logsm_obs, mu)
# plt.xlabel("logsm_obs")
# plt.ylabel("mu")
# plt.savefig("temp_mu_plot.pdf", dpi=300)

# plt.figure(figsize=(10,7))
# plt.hist(mu, bins=100, density=True)
# plt.savefig("temp_mu_hist.pdf", dpi=300)

# Stages:
#    1: project halo major axis and align in 2D
#        1.1: align with ellipticity = 1
#        1.2: align with ellipticity drawn from skysim
#    2: project halo major axis with full halo shape and align in 2D
#        2.1: align with ellipticity = 1
#        2.2: align with ellipticity drawn from skysim
#    3: align in 3D and project to 2D with full galaxy shape (Assume to be the same as halo)

##### STAGE 1 - DIRECT HALO MAJOR AXIS PROJECTION AND ALIGN IN 2D ##################################################################################################
print("STAGE 1...")
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
phi_aligned_2D_basic = get_position_angle(maj)

##### STAGE 2 - PROJECT HALO WITH FULL SHAPE AND ALIGN IN 2D TO RESULTING MAJOR AXIS ###############################################################################
print("STAGE 2...")
a = np.linalg.norm(halo_major_axis, axis=-1, keepdims=False)
b = np.linalg.norm(halo_inter_axis, axis=-1, keepdims=False) / a
c = np.linalg.norm(halo_minor_axis, axis=-1, keepdims=False) / a
a /= a
normed_major = normalized_vectors(halo_major_axis)
normed_inter = normalized_vectors(halo_inter_axis)
normed_minor = normalized_vectors(halo_minor_axis)
halo_phi_shape, halo_ellipticity_shape = compute_ellipse2d(a, b, c, halo_coords,
                                                     normed_major, normed_inter, normed_minor)

# Grab halo properties
halo_full_e1, halo_full_e2 = phi_to_e1_e2(halo_phi_shape, halo_ellipticity_shape)

# Align galaxies
phi_aligned_2D_full_shape = align_to_axis(halo_phi_shape, mu, as_vector=False, custom_distr=dwvm)

##### STAGE 1.1, 1.2, 2.1, 2.2 - ASSIGN ELLIPTICITIES AND GET E1,E2 RESULTS
# Get ellipticities from skysim shape distribution, assigning based on percentile
print("Assign e1, e2")

##### STAGE 1.1
e1_aligned_basic_one, e2_aligned_basic_one = phi_to_e1_e2(phi_aligned_2D_basic, 1.)

##### STAGE 1.2
percentiles = get_e_percentiles(halo_major_axis, halo_minor_axis, dataset)
mbd = MixedBetaDistribution()
params = {"w1":0.283, "alpha1":2.484, "beta1":14.896, "alpha2":2.174, "beta2":4.619}
mbd_ellipticity = mbd.ppf(percentiles, **params)
e1_aligned_basic_drawn, e2_aligned_basic_drawn = phi_to_e1_e2(phi_aligned_2D_basic, mbd_ellipticity)

##### STAGE 2.1
e1_aligned_shape_one, e2_aligned_shape_one = phi_to_e1_e2(phi_aligned_2D_full_shape, 1.)

##### STAGE 2.2
# percentiles have already been drawn, only need to do the assignment stage
e1_aligned_shape_drawn, e2_aligned_shape_drawn = phi_to_e1_e2(phi_aligned_2D_full_shape, mbd_ellipticity)

##### STAGE 3 - ALIGN IN 3D AND PROJECT GALAXIES TO 2D #############################################################################################################
print("STAGE 3...")
major, inter, minor = align_to_axis_3D(halo_major_axis, mu)
galaxy_phi_shape, galaxy_ellipticity_shape = compute_ellipse2d(a, b, c, coords,
                                                                major, inter, minor)

e1_galaxy_projected, e2_galaxy_projected = phi_to_e1_e2(galaxy_phi_shape, galaxy_ellipticity_shape)

# Save raw data (columns unaltered from the catalog), as well as calculated quantities (alignments, projected orientations, etc.)
f_name_base = "alignment_comparison"
np.savez(f"{f_name_base}_raw.npz", ra=ra, dec=dec, halo_phi=halo_phi, centrals=centrals, redshift=redshift, log_sm=logsm_obs)
np.savez(f"{f_name_base}_computed.npz", mu=mu, projected_halo_phi_basic=halo_phi, projected_halo_phi_full_shape=halo_phi_shape, 
                                        projected_galaxy_phi_full=galaxy_phi_shape,
                                        drawn_ellipticity=mbd_ellipticity, galaxy_projected_ellipticity=galaxy_ellipticity_shape,
                                        e1_full_halo=halo_full_e1, e2_full_halo=halo_full_e2,
                                        e1_basic_ellip1=e1_aligned_basic_one, e2_basic_ellip1=e2_aligned_basic_one,
                                        e1_basic_ellip_drawn=e1_aligned_basic_drawn, e2_basic_ellip_drawn=e2_aligned_basic_drawn,
                                        e1_full_shape_ellip1=e1_aligned_shape_one, e2_full_shape_ellip1=e2_aligned_shape_one,
                                        e1_full_shape_ellip_drawn=e1_aligned_shape_drawn, e2_full_shape_ellip_drawn=e2_aligned_shape_drawn,
                                        e1_galaxy_projected=e1_galaxy_projected, e2_galaxy_projected=e2_galaxy_projected)

# # Diagnostic plots
# x = np.linspace(0,1,1000)
# plt.plot(x, mbd.pdf(x, **params))
# plt.show()

print("Nothing crashed")