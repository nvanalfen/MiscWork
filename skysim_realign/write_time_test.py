import pandas as pd
import numpy as np
import time
import os
from astropy.io import ascii
from astropy.table import Table
from astropy.io.misc.hdf5 import write_table_hdf5

f_name = "ellipticity_add_on=0.0_1.0.dat"
data = ascii.read(os.path.join( "/pscratch", "sd", "v", "vanalfen", f_name ))

write_dir = os.path.join( "/pscratch", "sd", "v", "vanalfen", "skysim_realign" )

# create dataframe
data_dict = {key:data[key] for key in data.columns}
df = pd.DataFrame(data_dict)

print(len(df), flush=True)
print(df[:5], flush=True)
print(">>>", flush=True)
print(len(data), flush=True)
print(data[:5], flush=True)

print("\nWriting hdf5", flush=True)
start = time.time()
write_table_hdf5(data, os.path.join( write_dir, "hdf5_test.h5" ), overwrite=True)
print("Time: ",time.time()-start, flush=True)

print("\nWriting dat", flush=True)
start = time.time()
ascii.write(data, os.path.join( write_dir, "dat_test.dat" ), overwrite=True)
print("Time: ",time.time()-start, flush=True)

print("\nWriting CSV", flush=True)
start = time.time()
df.to_csv(os.path.join( write_dir, "csv_test.csv" ), index=False)
print("Time: ",time.time()-start, flush=True)