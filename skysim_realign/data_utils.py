import yaml
import numpy as np
import os
import h5py
import importlib
from modular_alignments.alignment_strengths import compound_sigmoid
from DW_to_VM_map import DimrothWatsonToVonMisesMapper

from modular_alignments.modular_alignment_2d import tidal_angle
from modular_alignments.modular_alignment import phi_to_e1_e2
from modular_alignments.modular_alignment import align_to_axis as align_to_axis_3d
from modular_alignments.modular_alignment_2d import align_to_axis as align_to_axis_2d
from modular_alignments.alignment_strengths import compound_sigmoid

def load_yaml_config(config_file, restructure=True):
    """
    Load a YAML configuration file.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def check_saved_config(config):
    """
    Check an existing file and compare the saved config.

    Parameters:
    -----------
    config:    dict config to compare to the stored config

    Returns:
    --------
    True if passed config perfectly matches stored config, False otherwise
    """
    f_name = config["output_file"]
    if not os.path.exists(f_name):
        return False
    # serialized_passed_config = serialize_config(config)
    with h5py.File(f_name, 'r') as f:
        if not "metadata" in f.keys() or not "config" in f["metadata"].keys():
            return False
        # Load config
        config_yaml = f['metadata/config'][()].decode('utf-8')
        h5_config = yaml.safe_load(config_yaml)
        if h5_config == config:
            return True
    return False

def save_hdf5(config, column_data, check_existing=True):
    f_name = config["output_file"]
    if check_existing and check_saved_config(config):
        print(f"File exists at {f_name} with the same config.\nSkipping save.", flush=True)
        return
    with h5py.File(f_name, "w") as f:
        f.create_group("metadata")
        f.create_group("data")

        metadata_group = f["metadata"]
        data_group = f["data"]

        # Store config
        config_content = yaml.dump(config, default_flow_style=False, sort_keys=False)
        metadata_group.create_dataset('config', data=config_content, dtype=h5py.string_dtype())

        # Store data columns
        for col in column_data:
            data_group.create_dataset(col, data=column_data[col])

def preprocess_config(config):
    """
    Preprocess the config and return true if a file at the designated location already exists, false otherwise
    """
    #############################################################################################################################################
    ##### ADJUST CONFIG #########################################################################################################################
    #############################################################################################################################################
    # Change the alignment_strength_complement section to match the alignment strength section (if use = True) in areas where it has no info
    if config.get("alignment_strength_complement", {}).get("use", False):
        inherit_alignment_config(config["alignment_strength"], config["alignment_strength_complement"])

    #############################################################################################################################################
    ##### CHECK EXISTING FILE ###################################################################################################################
    #############################################################################################################################################
    if check_saved_config(config):
        print("File exists with same config. Skipping", flush=True)
        return True
    return False

def gather_columns_and_filters(config):
    # Just the minimum relevant columns to align to tidal fields in 2D
    essential_columns = set([ "tidal_s_11", "tidal_s_12", "tidal_s_22", "redshiftHubble",
                        "morphology/totalEllipticity", "morphology/totalEllipticity1", "morphology/totalEllipticity2", "galaxyID"
                    ])
    extra_columns = list_extra_columns(config)
    table_columns = list(essential_columns | extra_columns)
    table_columns.sort()    # To avoid mismatched ordering when separate mpi ranks expect the same order of column data
    native_filters = [filt for filt in config["native_filters"]]
    filters = [filt for filt in config["filters"]]

    return table_columns, native_filters, filters

def inherit_alignment_config(primary_config, secondary_config):
    """
    Given a config, the default behavior for the alignment_strength_complement section is
    to inherit the config parameters in alignment_strength unless explicitly given.

    Parameters:
    -----------
    primary_config:    The part of the config dict defining parameters and behaviors.
    secondary_config:  The second part of the config into which we copy the primary_config values if none exists

    Returns:
    --------
    None, modify config in place
    """
    # For each key category in alignment_strength, it either maps to a dict or a value (number, string, list, etc.).
    # For all keys in alignment_strength, if alignment_strength_complement doesn't have it, simply copy all over.
    # If alignment_strength_complement has the key, traverse in and repeat.
    for key in primary_config:
        if isinstance(primary_config[key], dict):
            if not key in secondary_config:
                secondary_config[key] = {}
            inherit_alignment_config(primary_config[key], secondary_config[key])
        elif not key in secondary_config:
            secondary_config[key] = primary_config[key]

def is_lambda(expr):
    if not isinstance(expr, str):
        return False
    return "lambda" == expr.split(" ")[0]

def extract_requested_columns(config):
    extra_columns = set()
    for col in config["extra_columns"]:
        if col:
            extra_columns.add(col)
    return extra_columns

def extract_custom_columns(func_signature):
    extra_columns = set()
    if not func_signature or func_signature.strip() == "":
        # If there's a null or ampty criterion, ignore
        pass
    elif is_lambda(func_signature):
        extra_columns |= extract_lambda_columns(func_signature)
    else:
        func_name, func_args = func_signature.split("(")
        func_args = [ arg.strip() for arg in func_args.replace(")","").split(",") if arg != "" ]
        for arg in func_args:
            extra_columns.add(arg)
    return extra_columns

def extract_sigmoid_columns(sigmoid_chunk):
    extra_columns = set()
    lambdas = []
    # First, build a list of all lambda functions to parse
    for layer in sigmoid_chunk:
        # each layer is a sigmoid setup (name doesn't matter)
        for param in sigmoid_chunk[layer]:
            # Each parameter may be a lambda. At most one should be, but it can be different
            val = sigmoid_chunk[layer][param]
            if isinstance(val, str) and "lambda" == val.strip().split(" ")[0]:
                # If the value is a string where "lambda" is the first element, assume it's a lambda function and parse
                lambdas.append(val.strip())
    # Now, parse all lambdas and add the requested columns
    for expr in lambdas:
        args = extract_lambda_columns(expr)
        for arg in args:
            extra_columns.add(arg)
    return extra_columns

def extract_criteria_columns(criteria_chunk):
    """
    Get the columns necessary to measure user-defined sriteria for alignments.

    Parameters:
    -----------
    criteria_chunk:    Equivalent to config["alignment_strength"]["criteria"]. Contains a list of lambdas (or null)

    Returns:
    --------
    extra_columns:     Set of column names to load
    """
    extra_columns = set()
    for criterion in criteria_chunk:
        extra_columns |= extract_lambda_columns(criterion)
    return extra_columns

def extract_lambda_columns(expr, preserve_order=False):
    # extract the arg list between the lambda keyword and the function
    if not expr:
        return set()
    if not is_lambda(expr):
        print(f"Expected lambda, got {expr}. Skipping.", flush=True)
        return set()
    args = expr.replace("lambda","").strip().split(":")[0]
    # Clean the arg list
    args = [arg.strip() for arg in args.split(",") if arg != ""]
    if preserve_order:
        return args
    return set(args)

def perform_lambda(expr, data, mask=None):
    func = eval(expr)
    args = extract_lambda_columns(expr, preserve_order=True)  
    if mask is None:
        vals = [np.array(data[arg]) for arg in args]
    else:
        vals = [np.array(data[arg][mask]) for arg in args]
    return func(*vals)

def list_extra_columns(config):
    """
    Build a list of extra columns we need to grab from the catalog.
    """
    #     Extra columns include:
    # - extra_columns category in config
    # - if alignment_strength/alignment_strength_method = custom:
    #     - columns are arguments in alignment_strength/alignment_strength_method/custom
    #         - e.g. function_name(arg1, arg2, ...)
    # - if alignment_strength/alignment_strength_method = sigmoids:
    #     - columns are arguments in lambda expressions for each sigmoid in alignment_strength/alignment_strength_method/sigmoid/<label>/<attribute>
    #         - e.g. alignment_strength/alignment_strength_method/sigmoid/color/x: "lambda mag_true_g, mag_true_r : ..."
    extra_columns = set()

    # Grab all explicitly requested extra columns
    extra_columns |= extract_requested_columns(config)

    # For the main alignment strength section
    # Grab columns needed for custom scripts
    if config["alignment_strength"]["alignment_strength_method"] == "custom":
        extra_columns |= extract_custom_columns( config["alignment_strength"]["custom"]["alignment_strength"] )
    
    # Grab columns needed for sigmoids
    if config["alignment_strength"]["alignment_strength_method"] == "sigmoid":
        extra_columns |= extract_sigmoid_columns(config["alignment_strength"]["sigmoid"])

    # Grab columns for alignment mask criteria
    if config["alignment_strength"].get("criteria",None):
        extra_columns |= extract_criteria_columns(config["alignment_strength"]["criteria"])

    # For the alignment strength complement_section
    # Only do if use = True and the proper sections exist, otherwise no new columns are needed
    if config["alignment_strength_complement"]["use"] and config["alignment_strength_complement"]["alignment_strength_method"]:
        if config["alignment_strength_complement"]["alignment_strength_method"] == "custom":
            extra_columns |= extract_custom_columns( config["alignment_strength_complement"]["custom"]["alignment_strength"] )
        if config["alignment_strength_complement"]["alignment_strength_method"] == "sigmoid":
            extra_columns |= extract_sigmoid_columns(config["alignment_strength_complement"]["sigmoid"])
        if config["alignment_strength_complement"].get("criteria",None):
            extra_columns |= extract_criteria_columns(config["alignment_strength_complement"]["criteria"])
    
    return extra_columns

def get_alignment_masks(config, data):
    """
    Get the masks to align e1 and e2. Returns two masks, once for the primary criteria, one for the complementary set.

    Parameters:
    -----------
    config:         The config for the current run. Contains the two alignment categories
    data:           The catalog data with columns for determining the masks.

    Returns:
    primary_mask:   Mask satisfying the primary alignment criteria
    secondary_mask: Mask satisfying the secondary criteria (complement of primary by default)
    """
    keys = list(data.keys())
    primary_mask = np.ones(len(data[keys[0]]), dtype=bool)
    secondary_mask = np.zeros(len(primary_mask), dtype=bool)

    # Get primary mask
    criteria = config["alignment_strength"].get("criteria",None)
    if criteria:
        for criterion in criteria:
            if criterion:
                primary_mask &= np.array( perform_lambda(criterion, data) )

    # Get secondary mask
    # By default, Case 1: all zeros
    complement = config["alignment_strength_complement"]
    if complement["use"]:
        criteria = complement.get("criteria",[])
        criteria = list(filter(None, criteria))        # Remove null criteria from list (there should always be one if you left the list in the config intact)
        if criteria and len(criteria) > 0:
            # Case 2a: Explicit criteria
            for criterion in criteria:
                if criterion:
                    secondary_mask &= np.array( perform_lambda(criterion, data) )
        else:
            # Case 2b: simple complement of primary mask
            # Simply the complement if use==True but no criteria set
            secondary_mask = ~primary_mask

    return primary_mask, secondary_mask

def calculate_custom_mu(data, mask, custom_script, custom_func):
    if is_lambda(custom_func):
        func = eval(custom_func)
        func_args = extract_lambda_columns(custom_func, preserve_order=True)
    else:
        func_name, func_args = custom_func.split("(")
        func_args = [ arg.strip() for arg in func_args.replace(")","").split(",") if arg!= "" ]
        try:
            module = importlib.import_module(custom_script)
            func = getattr(module, func_name.strip())
        except ImportError:
            raise ImportError(f"Function {func_name} not found in {custom_script}")

    args = [ np.array(data[arg][mask]) for arg in func_args ]
    mus = func(*args)
    return mus

def calculate_sigmoid_mu(data, mask, sigmoid_section):
    """
    Calculates mu values from sigmoid parameters given. With multiple sigmoids, they are chained together.

    Parameters:
    -----------
    data:               The catalog data containing columns to be used in calculating alignment strength
    mask:               The mask designating the section of data to use
    sigmoid_section:    The sigmoid section of the config

    Returns:
    --------
    mus calculated from the chained sigmoids.
    """
    xs = []
    params = []
    layered_parameter = []
    ind = 0
    max_ind = len(sigmoid_section)

    for _, layer in sigmoid_section.items():
        current_params = {}
        for param in layer:
            # Grab the value at the given parameter
            value = layer[param]
            if is_lambda(value):
                # If the value is a lambda expression, turn it into a number or iterable
                value = perform_lambda(value, data, mask)

            # Store the value properly
            if param == "x":
                xs.append(value)
            elif param == "use_as":
                if ind == max_ind-1:
                    # Don't append the use_as for the final sigmoid
                    continue
                if not value:
                    value = "y_high"
                layered_parameter.append(value)
            else:
                current_params[param] = value

        params.append(current_params)
        ind += 1

    return compound_sigmoid(xs, params, layered_parameter)

def calculate_mus(config, data, primary_mask, secondary_mask):
    """
    Calculate the array of mu values to be used for realigning.
    Zeros by default, overwrite on the masks.

    Parameters:
    -----------
    config:            Config settings (containing methods to calculate mu)
    data:              The catalog data
    primary_mask:      The mask to determine which galaxies get the primary alignment
    secondary_mask:    The mask to determine which galaxies get the secondary alignment

    Returns:
    --------
    mus:               Array of mu values
    """

    mus = np.zeros(len(primary_mask))

    # Extract the variables (cleaner to do it here than in each if)
    primary_method = config["alignment_strength"]["alignment_strength_method"]
    primary_constant_section = config["alignment_strength"].get("constant", {})
    primary_custom_section = config["alignment_strength"].get("custom", {})
    primary_sigmoid_section = config["alignment_strength"].get("sigmoid", {})
    use_complement = config.get("alignment_strength_complement", {}).get("use", False)
    if use_complement:
        # Default to inheriting the primary methods (it would be silly, since why would you need two versions, but it's default)
        secondary_method = config["alignment_strength_complement"].get("alignment_strength_method", primary_method)
        secondary_constant_section = config["alignment_strength_complement"].get("constant", primary_constant_section)
        secondary_custom_section = config["alignment_strength_complement"].get("custom", primary_custom_section)
        secondary_sigmoid_section = config["alignment_strength_complement"].get("sigmoid", primary_sigmoid_section)

    # Calculate primary
    if primary_method == "constant":
        mus[primary_mask] = primary_constant_section["alignment_strength"]
    elif primary_method == "custom":
        custom_script = primary_custom_section.get("custom_script",None)
        custom_func = primary_custom_section["alignment_strength"]
        mus[primary_mask] = calculate_custom_mu(data, primary_mask, custom_script, custom_func)
    elif primary_method == "sigmoid":
        mus[primary_mask] = calculate_sigmoid_mu(data, primary_mask, primary_sigmoid_section)
    else:
        method = config["alignment_strength"]["alignment_strength_method"]
        print(f"Alignment strength method \"{method}\" not supported.", flush=True)

    # Calculate secondary
    if use_complement:
        if secondary_method == "constant":
            mus[secondary_mask] = secondary_constant_section["alignment_strength"]
        elif secondary_method == "custom":
            custom_script = secondary_custom_section.get("custom_script",None)
            if not custom_script:
                custom_script = primary_custom_section.get("custom_script",None)
            if not custom_script:
                print("No custom script specified for complement alignment strength.", flush=True)
            custom_func = secondary_custom_section["alignment_strength"]
            mus[secondary_mask] = calculate_custom_mu(data, secondary_mask, custom_script, custom_func)
        elif secondary_method == "sigmoid":
            mus[secondary_mask] = calculate_sigmoid_mu(data, secondary_mask, secondary_sigmoid_section)
        else:
            method = config["alignment_strength_complement"]["alignment_strength_method"]
            print(f"Alignment strength method \"{method}\" not supported.", flush=True)
    return mus

def mu_map_generic(mu, *args):
    args = [*args, 0]           # Force mu=0 to map to 0
    return np.polyval(args, mu)

def mu_map_generic_free(mu, *args):
    return np.polyval(args, mu)

def remap_mu_tidal(mus, mask, remap_tidal):
    """
    Remap the calculated mus in the mask if requested.

    Parameters:
    -----------
    mus:          array[float], Array of mu values to remap.
    mask:         array[bool], The mask picking out the relevant mu values.
    remapping:    bool, Whether to remap. If True, assume the given mu is the equivalent value for 3D alignment w.r.t. halos,
                  meaning we need to remap to 2D tidal.

    Returns:
    None, modify in place
    """
    if remap_tidal:
        # Very simple, obtained via basic fitting
        tidal_remap = 0.67
        mus[mask] *= tidal_remap

def remap_mu_3d(remap_3d):
    """
    Return the model to map a single mu for Dimroth-Watson to the equivalent parameters
    for a weighted average of two von Mises distributions.

    Parameters:
    -----------
    remap_3d:    bool, whether to remap (assume the given mu is the value for aligning in 3D)
    """
    remap_model = None
    if remap_3d:
        primary_vm_params = [-0.20252838, 0.10862449, 1., -0.08426155, 0.11907565]
        secondary_vm_params = [0.020614, 0.05596696]
        weight_params = [0.08546417, 0.06449063, 0.08947536]
        primary_vm_mapper = mu_map_generic
        secondary_vm_mapper = mu_map_generic
        weight_mapper = mu_map_generic_free
        remap_model = DimrothWatsonToVonMisesMapper(primary_vm_params=primary_vm_params, secondary_vm_params=secondary_vm_params, 
                                             weight_params=weight_params, primary_vm_mapper=primary_vm_mapper, 
                                             secondary_vm_mapper=secondary_vm_mapper, weight_mapper=weight_mapper)
    return remap_model

def align(data, mus, primary_mask, secondary_mask, primary_remap_model, secondary_remap_model):
    """
    Align primary and 
    """
    sxx = data["tidal_s_11"]
    sxy = data["tidal_s_12"]
    syy = data["tidal_s_22"]
    redshift = data["redshiftHubble"]
    tidal_phi = tidal_angle(sxx, syy, sxy, redshift)
    mu = np.ones(len(sxx))*mus
    primary_aligned_phi = align_to_axis_2d(tidal_phi[primary_mask], mu[primary_mask], as_vector=False, custom_distr=primary_remap_model)
    secondary_aligned_phi = align_to_axis_2d(tidal_phi[secondary_mask], mu[secondary_mask], as_vector=False, custom_distr=secondary_remap_model)

    return primary_aligned_phi, secondary_aligned_phi

def rotate_shapes(data, phi_primary, phi_secondary, primary_mask, secondary_mask):
    ellipticity = data["morphology/totalEllipticity"]
    primary_e1, primary_e2 = phi_to_e1_e2(phi_primary, ellipticity[primary_mask])
    secondary_e1, secondary_e2 = phi_to_e1_e2(phi_secondary, ellipticity[secondary_mask])
    return primary_e1, primary_e2, secondary_e1, secondary_e2
    