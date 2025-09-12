import os
import h5py
import numpy as np

# Get MPI stuff
from mpi4py import MPI
comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

import data_utils
import skysim_realign

def copy_h5_data(grp, data, upper_key=None):
    for key in grp.keys():
        new_key = key
        if upper_key:
            new_key = "/".join([upper_key, key])
            
        if isinstance(grp[key], h5py.Group):
            copy_h5_data(grp[key], data, upper_key=new_key)
        else:
            if not new_key in data:
                data[new_key] = []
            data[new_key].append( grp[key][:] )

def convert_h5_array(data):
    for key in data:
        if isinstance(data[key], dict):
            convert_h5_array(data[key])
        else:
            data[key] = np.hstack(data[key])

def merge_h5(base_f_name, base_config, filters, extra_native_filters, clean=False):
    merged_data = {}

    # Load data from subset files
    for i in range(len(filters)):
        with h5py.File(base_f_name.format(i), "r") as f:
            data_grp = f["data"]
            copy_h5_data(data_grp, merged_data)

    # Convert the lists in the data dict to single arrays
    convert_h5_array(merged_data)

    merged_f_name = base_f_name.format("merged")
    base_config["output_file"] = merged_f_name
    data_utils.save_hdf5(base_config, merged_data, check_existing=False)

    # Append extra filters
    with h5py.File(merged_f_name, "a") as f:
        f["metadata"].create_group("extra_filters")
        filter_grp = f["metadata/extra_filters"]
        for i, filts in enumerate(extra_filters):
            filt_list = np.array(filts, dtype=h5py.string_dtype(encoding='utf-8'))
            filter_grp.create_dataset(f"filters_{i}", data=filt_list)

    if clean:
        for i in range(len(filters)):
            os.remove(base_f_name.format(i))

###########################################################################################################################################
##### VARIABLE INPUT ######################################################################################################################
###########################################################################################################################################
merge_final = True
config_f_name = "base_config.yaml"
extra_native_filters = [
    ["redshift_block_lower <= 0"],
    ["redshift_block_lower <= 0"],
    ["redshift_block_lower <= 1"],
    ["redshift_block_lower <= 1"],
    ["redshift_block_lower <= 2"],
    ["redshift_block_lower <= 2"]
]
extra_filters = [
    ["redshiftHubble < 0.5"],
    ["redshiftHubble >= 0.5", "redshiftHubble < 1.0"],
    ["redshiftHubble >= 1.0", "redshiftHubble < 1.5"],
    ["redshiftHubble >= 1.5", "redshiftHubble < 2.0"],
    ["redshiftHubble >= 2.0", "redshiftHubble < 2.5"],
    ["redshiftHubble >= 2.5"]
]
###########################################################################################################################################
##### END VARIABLES #######################################################################################################################
###########################################################################################################################################

config = None
if rank == 0:
    config = data_utils.load_yaml_config(config_f_name)

# Broadcast the config to all ranks
config = comm.bcast(config, root=0)
base_output_f_name = config["output_file"]
base_native_filters = config["native_filters"]
base_filters = config["filters"]

# Loop through each filter, adjust the config, and do our thing
for i in range(len(extra_filters)):
    if rank == 0:
        print(f"Initiating loop {i}", flush=True)
    config["output_file"] = base_output_f_name.format(i)
    config["native_filters"] = extra_native_filters[i]
    config["filters"] = list(base_filters)
    for filt in extra_filters[i]:
        config["filters"].append(filt)

    skysim_realign.main(config)
    comm.Barrier()                           # Block all ranks until the full run is done before going on to the next

# Merge resulting h5 files
if rank == 0:
    # Restore base config
    config["native_filters"] = base_native_filters
    config["filters"] = base_filters
    config["output_file"] = base_output_f_name
    merge_h5(base_output_f_name, config, extra_filters, extra_native_filters)