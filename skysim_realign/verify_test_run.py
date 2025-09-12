import numpy as np
import h5py
from modular_alignments.alignment_strengths import sigmoid, compound_sigmoid

result_f_name = "test.h5"
data_f_name = "super_mini_subset.h5"

def test_mu():
    primary_mask = np.array([1,0,1,1,1,0,1,1,1,1], dtype=bool)
    secondary_mask = ~primary_mask
    color_params = {"x0":4, "k":-0.4,"y_low":-0.2,"y_high":1.}
    mag_params = {"x0":28,"k":-0.1,"y_low":-0.2,"y_high":1.}
    
    mus = np.zeros(len(primary_mask))

    result_data = h5py.File(result_f_name, "r")
    subset_data = h5py.File(data_f_name, "r")

    G = subset_data["mag_true_g"][:]
    R = subset_data["mag_true_r"][:]
    results = sigmoid(G-R, **color_params)
    mag_params["y_high"] = results
    results = sigmoid(R, **mag_params)

    mus[primary_mask] = results[primary_mask]        # Explicitly do NOT remap these. Mostly for testing sake
    mus[secondary_mask] = 0.9*0.67

    assert all(result_data["data/mu"][:] == mus)

    result_data.close()
    subset_data.close()
    print("PASSED TEST MU")

if __name__ == "__main__":
    test_mu()

# # Assuming the following data...
# data_dict = {
#     "galaxyID" : np.array([10048000243153, 10048000243159, 10048000243164, 10048000243167,
#    10048000243174, 10048000243180, 10048000243188, 10048000243195,
#    10048000243197, 10048000243198]),
#     "mag_true_g" : np.array([23.004826, 22.331701, 24.339354, 25.303   , 25.436428, 22.435316,
#    24.76393 , 23.631788, 23.581116, 24.353907], dtype=np.float32),
#     "mag_true_r" : np.array([21.382687, 21.884518, 22.761456, 23.79417 , 24.002644, 21.98813 ,
#    23.096352, 22.932407, 23.09235 , 22.712307], dtype=np.float32),
#     "redshiftHubble" : np.array([0.80099058, 0.80099058, 0.80099058, 0.80099058, 0.80099058,
#    0.80099058, 0.80099058, 0.80099058, 0.80099058, 0.80099058]),
#     "tidal_s_11" : np.array([1.51219788, 1.44666211, 1.63791055, 1.44231371, 1.00599651,
#    1.50644213, 1.53956033, 1.17460474, 1.33472185, 1.64632943]),
#     "tidal_s_12" : np.array([0.60851702, 0.61266809, 0.61130985, 0.60388418, 0.72985301,
#    0.60552587, 0.64293075, 0.56982937, 0.60578241, 0.61018375]),
#     "tidal_s_22" : np.array([1.13943989, 1.05196306, 1.26268351, 1.08247184, 0.54165417,
#    1.05228174, 1.10618964, 0.8346766 , 0.93343482, 1.27541581]),
#     "morphology/totalEllipticity" : np.array([0.20381795, 0.08824219, 0.32843938, 0.37253517, 0.12894392,
#    0.1846683 , 0.18366903, 0.04851675, 0.49389598, 0.04180954], dtype=np.float32),
#     "morphology/totalEllipticity1" : np.array([-0.17713988, -0.08819129,  0.25685507,  0.26360357, -0.03368311,
#    -0.15881163,  0.04467028,  0.02475647,  0.27476323,  0.03480506], dtype=np.float32),
#     "morphology/totalEllipticity2" : np.array([-0.1008128 , -0.00299667,  0.20468977, -0.2632406 , -0.12446679,
#    -0.09424038, -0.17815408,  0.0417252 , -0.4104125 , -0.02316561], dtype=np.float32)
# }
# data = Table(data_dict)
# # run the skysim_realign.py file with test = true in the config. Then this will test that

# The following is the test_config.py file used to generate the data being tested
"""
output_file: "/pscratch/sd/v/vanalfen/skysim_realign/add_ons/ellipticity_add_on.h5"
test: True
overwrite_match: True
native_filters:
    - "redshift_block_lower <= 0"
filters:
    - "redshiftHubble >= 0.8"
    - "redshiftHubble < 0.81"
    - "mag_true_r_lsst < 24.5"
    - "hostHaloMass > 1.846e13"      # same as baseDC2/target_halo_mass
sparse_storage: true                  # true: store only realigned e1,e2. false: store all e1, e2
downsample_fraction: 1.               # What fraction to keep [0,1] where 1.0 = no downsampling, 0.5 = randomly keep half, 0 = keep none
extra_columns:                        # Extra columns to store in the output
    # By default, store only [galaxyID, e1, e2, mu]. Add extras if you want to store more below
    - null                            # Keep this line. Will not add a column, but will allow the config to read this section as legitimate

alignment_strength:
    # Criteria creates a mask and only applies alignment strengths to those satisfying the criteria given as lambdas
    # With multiple criteria, each resulting mask will be ANDed together
    # if null, this applies to ALL galaxies read in
    criteria:
        - null                        # Keep this line. Add extras below to add conditions
        - "lambda mag_true_g : mag_true_g > 23"
        #- "lambda baseDC2/target_halo_mass: baseDC2/target_halo_mass > 1.846e13"
    # alignment_method options: halo, tidal, sigmoid, custom
    #    halo:       Align with respect to halo (uses major axis)
    #    tidal:      Align with respect to tidal fields (uses tidal tensors)
    #    Recommendation: use tidal and simply set remap_tidal to true (and remap_3D to true if assuming alignment calculated in 3D)
    alignment_method: "tidal"
    # alignment_strength_method options: constant, custom, sigmoid
    #    constant:   A single number to use as the alignment strength for all
    #    custom:     A function call of the form "func_name(arg1, arg2,...)"
    #                    - func_name: the name of a custom function (in a separate script that will be imported using a parameter below)
    #                    - arg1, arg2, ...: column names in the loaded catalog
    #                Alternately, a lambda function "lambda arg1, arg2 : function_logic"
    #                    - arg1, arg2: column names in the loaded table
    #    sigmoid:    A series of "layers" (their names don't matter). Each with the parameters needed to define a sigmoid.
    #                    The parameters will be either a value, or a lambda function accessing the table columns
    alignment_strength_method: "sigmoid"
    # remap_mu options: null, tidal, 3d
    #     null:      No remapping. Use the mus as given/calculated (assume the user has passed them in the right frame)
    #     tidal:     Assume the given mus are appropriate for 3D alignments w.r.t. halos. Remap to 2D tidal.
    #     2d or 2D:  Assume the given mus are appropriate for 3D alignments w.r.t. halos. Remap to 2D halo (not yet fully supported).
    remap_3d: True
    remap_tidal: False
    constant:
        # Only applies if constant method chosen
        alignment_strength: 0.9    # Alignment strength
    custom:
        # Only applies if custom method chosen - set custom_script to null if using a lambda function for alignment strength
        custom_script: "custom_scripts.py"               # python script containing functions to use to assign individual alignment strengths
        alignment_strength: "color_alignment(mag_true_r, mag_true_g)"    # Calling the function with column names
        # Alternatively, you could use a lambda. "lambda mag_true_r, mag_true_g: some_code..."
    sigmoid:
        # Only applies if sigmoid method chosen
        # Give a series of sigmoid parameters (either numerical, special function, or columns)
        # For each sigmoid, create a new indented category (name doesn't matter except for your understanding)
        # Multiple sigmoids will build on each other (i.e. use the results of the previous as a parameter in the next)
        # Example:
        color:
            # Sigmoid equation: y_low + (y_high - y_low) / (1 + exp[-k * [x - x0]])
            x: "lambda mag_true_r, mag_true_g: mag_true_g - mag_true_r"
            x0: 4
            k: -0.4
            y_low: -0.2
            y_high: 1.
            use_as: null            # If null, assume y_high. The parameter that these results will be used as in the following sigmoids
        luminosity:
            x: "lambda mag_true_r: mag_true_r"
            x0: 28
            k: -0.1
            y_low: -0.2
            y_high: 1.              # This will be overwritten since the last sigmoid results take its place
            use_as: null
            
alignment_strength_complement:
    # apply alignments to galaxies fulfilling other crietria
    # e.g. in the case of bad halos, the alignment_Strength section will align high mass halos and this section can be used
    # to align all galaxies in low mass halos differently, like w.r.t. tidal fields
    # CAUTION: if criteria overlap, this will overwrite previously realigned e1,e2 since it comes last
    use: true                     # Whether to use this section. if null, assume false
    criteria:
        - null                 # New criteria. If null, simply use the complement of the alignment_strength mask
    alignment_strength_method: "constant"
    remap_3d: True
    remap_tidal: True
    # For all other sections, assume the same values as alignment_strength section where they are not explcitly given
    # i.e. only include sections here that you wish to act differently than above. Otherwise, use the same settings as
    # from the alignment_strength section
    # Example usage:
    # in alignment_strength, set criteria for above a given mass, align those galaxies with respect to halos, set remap_3D to true
    # and remap_tidal to false, and whatever other settings we want. Here, set use to true, criteria to null (the complement), set
    # remap_tidal to true, and leave out all other fields (to automatically inherit)
"""