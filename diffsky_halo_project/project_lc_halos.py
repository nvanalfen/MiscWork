import numpy as np
import os
from halotools.utils.mcrotations import random_perpendicular_directions
from halotools.utils import elementwise_dot, angles_between_list_of_vectors, normalized_vectors
from ellipse_proj_dumb import compute_ellipse2d_dumb
import opencosmo as oc
import time
import h5py
import yaml
from pathlib import Path

def write_error(f_name, err):
    file_path = "error_lc.txt"
    try:
        with open(file_path, 'a') as file:
            file.write(f"\n\n>>>>> Error on file {f_name}\n")
            file.write(str(err))
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"A file write error occurred: {e}")

def zero_mask(vec):
    nulls = np.ones(len(vec)).astype(bool)
    if len(vec.shape) > 1:
        for i in range(vec.shape[1]):
            nulls &= vec[:,i] == 0
    else:
        nulls &= vec == 0
    return nulls

def nan_mask(vec):
    nans = np.zeros(len(vec)).astype(bool)
    if len(vec.shape) > 1:
        for i in range(vec.shape[1]):
            nans |= np.isnan(vec[:,i])
    else:
        nans |= np.isnan(vec)
    return nans

def sum_zero_vec(vec):
    nulls = zero_mask(vec)
    return sum(nulls)

def bad_mask(vec):
    return zero_mask(vec) | nan_mask(vec)

def check_bad_axes(pos, a, b, c, major, inter, minor):
    print("Nan Counts:")
    print(f"pos: {sum(np.isnan(pos))}")
    print(f"a: {sum(np.isnan(a))}")
    print(f"b: {sum(np.isnan(b))}")
    print(f"c: {sum(np.isnan(c))}")
    print(f"major: {sum(np.isnan(major))}")
    print(f"inter: {sum(np.isnan(inter))}")
    print(f"minor: {sum(np.isnan(minor))}")

    print("\nZero Counts:")
    print(f"pos: {sum_zero_vec(pos)}")
    print(f"a: {sum_zero_vec(a)}")
    print(f"b: {sum_zero_vec(b)}")
    print(f"c: {sum_zero_vec(c)}")
    print(f"major: {sum_zero_vec(major)}")
    print(f"inter: {sum_zero_vec(inter)}")
    print(f"minor: {sum_zero_vec(minor)}")

def replace_bad_axes(major, inter, minor, mask):
    pass

def project_single_lc_core_file(sim_dir, mock_file, batch_size, save_f_name, print_progress):
    dataset = oc.open(os.path.join(sim_dir, mock_file))
    N = len(dataset)
    dataset = dataset.take(N, at="start")

    cols = ["top_host_infall_fof_halo_eigS1X", "top_host_infall_fof_halo_eigS1Y", "top_host_infall_fof_halo_eigS1Z",
           "top_host_infall_fof_halo_eigS2X", "top_host_infall_fof_halo_eigS2Y", "top_host_infall_fof_halo_eigS2Z",
           "top_host_infall_fof_halo_eigS3X", "top_host_infall_fof_halo_eigS3Y", "top_host_infall_fof_halo_eigS3Z",
           "x","y","z"]

    data = dataset.select(cols).get_data("numpy")

    major_axis = np.array([data["top_host_infall_fof_halo_eigS1X"], data["top_host_infall_fof_halo_eigS1Y"], data["top_host_infall_fof_halo_eigS1Z"]]).T
    inter_axis = np.array([data["top_host_infall_fof_halo_eigS2X"], data["top_host_infall_fof_halo_eigS2Y"], data["top_host_infall_fof_halo_eigS2Z"]]).T
    minor_axis = np.array([data["top_host_infall_fof_halo_eigS3X"], data["top_host_infall_fof_halo_eigS3Y"], data["top_host_infall_fof_halo_eigS3Z"]]).T
    pos = np.array([data["x"],data["y"],data["z"]]).T

    # Replace bad axes with randoms
    axis_mask = bad_mask(major_axis) | bad_mask(inter_axis) | bad_mask(minor_axis)
    pos_mask = bad_mask(pos)
    mask = pos_mask | axis_mask

    a = np.linalg.norm(major_axis, axis=-1)
    b = np.linalg.norm(inter_axis, axis=-1)
    c = np.linalg.norm(minor_axis, axis=-1)
    major_normed = normalized_vectors(major_axis)
    inter_normed = normalized_vectors(inter_axis)
    minor_normed = normalized_vectors(minor_axis)

    # Project halos
    start_ind = 0
    batches = int(N/batch_size)+1
    
    alpha = np.zeros(len(a))
    beta = np.zeros(len(a))
    e_alpha = np.zeros(shape=(len(a),2))
    e_beta = np.zeros(shape=(len(a),2))
    
    for i in range(batches):
        if print_progress:
            print(f"{start_ind}-{start_ind+batch_size} / {N}")
        start = time.time()
        sub_mask = ~mask[start_ind:start_ind+batch_size]
        ellipse2d = compute_ellipse2d_dumb(a[start_ind:start_ind+batch_size][sub_mask], b[start_ind:start_ind+batch_size][sub_mask], 
                                           c[start_ind:start_ind+batch_size][sub_mask],
                                           pos[start_ind:start_ind+batch_size][sub_mask], 
                                           major_normed[start_ind:start_ind+batch_size][sub_mask], inter_normed[start_ind:start_ind+batch_size][sub_mask], 
                                           minor_normed[start_ind:start_ind+batch_size][sub_mask])
        # e_alpha.append( ellipse2d["e_alpha"] )
        # e_beta.append( ellipse2d["e_beta"] )
        # beta.append( ellipse2d["beta"] )
        # alpha.append( ellipse2d["alpha"] )
        alpha[start_ind:start_ind+batch_size][sub_mask] = ellipse2d["alpha"]
        beta[start_ind:start_ind+batch_size][sub_mask] = ellipse2d["beta"]
        e_alpha[start_ind:start_ind+batch_size][sub_mask] = ellipse2d["e_alpha"]
        e_beta[start_ind:start_ind+batch_size][sub_mask] = ellipse2d["e_beta"]
        if print_progress:
            print("Time = ",time.time()-start)
        
        start_ind += batch_size

    with h5py.File(save_f_name, 'a') as f:
        if not sim_dir in f:
            f.create_group(sim_dir)
        group = f[sim_dir]
    
        if not mock_file.replace(".hdf5","") in group.keys():
            group.create_group(mock_file.replace(".hdf5",""))
        group = group[mock_file.replace(".hdf5","")]
    
        alpha_set = group.create_dataset("alpha", alpha.shape, dtype=alpha.dtype)
        beta_set = group.create_dataset("beta", beta.shape, dtype=beta.dtype)
        e_alpha_set = group.create_dataset("e_alpha", e_alpha.shape, dtype=e_alpha.dtype)
        e_beta_set = group.create_dataset("e_beta", e_beta.shape, dtype=e_beta.dtype)
        alpha_set[:] = alpha
        beta_set[:] = beta
        e_alpha_set[:] = e_alpha
        e_beta_set[:] = e_beta

def project_all_lc_core_files(config_f_name):
    with open(config_f_name, 'r') as file:
        config = yaml.safe_load(file)

    main_dir = config["main_dir"]
    sim = config["sim"]
    sim_dir = os.path.join(main_dir, sim)
    all_files = config.get("all_files", True)
    batch_size = config["batch_size"]
    print_progress = config["print_progress"]
    save_file = config["save_file"]

    files = []
    if not all_files:
        files = [config["mock_file"]]
    else:
        data_path = Path(sim_dir)
        files = list(f for f in data_path.glob("*.hdf5") if f.stem.startswith("lc_cores"))

    for file in [str(f) for f in files]:
        # skip if the data exists (i.e. if a group with the file name exists)
        g_name = file.replace(".hdf5","")
        if os.path.exists(save_file):
            with h5py.File(save_file, 'r') as f:
                skip = False
                if sim_dir in f:
                    if g_name in f[sim_dir]:
                        skip = True
        if skip:
            print(f"{file} already done")
            continue
        print(f"Projecting {file}")

        try:
            project_single_lc_core_file(sim_dir, file, batch_size, save_file, print_progress)
        except Exception as e:
            print(f">>>>> ERROR: {g_name}")
            write_error(file, e)

def main(config_file="config.yaml"):
    project_all_lc_core_files(config_file)

if __name__ == "__main__":
    main()