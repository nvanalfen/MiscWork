# Running Single
To run realignments (properly):

- Edit config.yaml with the parameters desired for the run

- Activate the proper environment (copy-paste friendly given below)</br>
```
salloc -N 4 --qos interactive --time 03:00:00 --constraint cpu --account=m1727
source /global/common/software/lsst/common/miniconda/setup_current_python.sh
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
```

- Run </br>
```
srun -n 32 python3 skysim_realign.py config.yaml
```

# Run In Slices
For memory, a full run is not possible in one go. In these cases, we will want to run in narrower redshift slices (or any form of slice).

- Edit config.yaml
    - Change `output_file` to a formattable name (i.e. containing a `{}`)
        - e.g. `output_file: add_on_{}.h5`
    - Remove the filters you will traverse from the main config
        - e.g. for redshift slices, remove any `redshiftHubble > 0.5`, `redshiftHubble <= 1.0` from the config

- Edit variable input section of `wrapper_skysim_realign.py`
    - Set `merge_final` to `True` if you want the separate h5 files to be merged in the end
    - Set the value of `config_f_name` to the config file
    - Set the values of `extra_filters` to the filters delineating the slices
        - Each element of `extra_filters` is a list where each element is a filter
            - e.g. `extra_filters = [ ["redshiftHubble < 0.5"], ["redshiftHubble >= 0.5", "redshiftHubble < 1.0"] ]`

- Activate environment</br>
```
salloc -N 4 --qos interactive --time 03:00:00 --constraint cpu --account=m1727
source /global/common/software/lsst/common/miniconda/setup_current_python.sh
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
```

- Run</br>
```
srun -n 32 python3 wrapper_skysim_realign.py
```