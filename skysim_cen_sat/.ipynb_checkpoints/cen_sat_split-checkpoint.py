import numpy as np
from astropy.table import Table
from astropy.io import ascii
import h5py
import yaml
import sys
import time
import os

F_NAME_TEMPLATE = "/global/cfs/cdirs/lsst/groups/WL/projects/wl-massmap/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/GalCat_tomo{}_06GpAM_new.dat"
TOTAL_F = 5

def load_yaml(yaml_name='config.yaml'):
    try:
        with open(yaml_name, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print("Error: config.yaml not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")

def read_table(f_name):
    tab = ascii.read(f_name)
    return tab

def get_cen_mask(table, col_name="is_central"):
    return table[col_name] == "True"

def split_table(table, col_name="is_central"):
    mask = get_cen_mask(table, col_name)
    return table[mask], table[~mask]

def store_h5(f_name, cen_table, sat_table, label):
    with h5py.File(f_name, "a") as f:
        f.create_group(label)
        f[label].create_group("centrals")
        f[label].create_group("satellites")
        cen_group = f[f"{label}/centrals"]
        sat_group = f[f"{label}/satellites"]
        for col in cen_table.columns:
            if col == "is_central":
                # We have already split based on this, no need to store
                continue
            cen_group.create_dataset(col, data=cen_table[col].value)
        for col in sat_table.columns:
            if col == "is_central":
                continue
            sat_group.create_dataset(col, data=sat_table[col].value)

if __name__ == "__main__":
    config_f_name = "config.yaml"
    if len(sys.argv) > 1:
        config_f_name = sys.argv[1]
    config = load_yaml(config_f_name)

    f_name_template = config.get("f_name_template", F_NAME_TEMPLATE)
    n_tomo = config.get("n_tomo", 0)
    output_loc = config.get("output", "output.h5")

    print(f"{n_tomo} Tomographic bins")
    print(f"Will read files:\n {f_name_template}")
    print(f"And will store at {output_loc}")

    directory = os.path.dirname(output_loc)
    os.makedirs(directory, exist_ok=True)

    for i in range(n_tomo):
        print(f">>>>> Tomo bin {i+1}")
        print("Loading Table")
        start = time.time()
        tab = read_table(f_name_template.format(i+1))
        tot = time.time()-start
        print(f">> {tot}")
        print("Splitting Table")
        start = time.time()
        cen_tab, sat_tab = split_table(tab)
        tot = time.time()-start
        print(f">> {tot}")
        print("Storing Data")
        start = time.time()
        store_h5(output_loc, cen_tab, sat_tab, f"tomo_{i+1}")
        tot = time.time()-start
        print(f">> {tot}")