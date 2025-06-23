import h5py
import os

f_name = "ellipticity_add_on.h5"
full_f_name = os.path.join( "/pscratch", "sd", "v", "vanalfen", "skysim_realign", "add_ons", f_name )

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(f"File: {full_f_name}")
print(f"Exists {os.path.exists(full_f_name)}")

if os.path.exists(full_f_name):
    total = 0
    with h5py.File(full_f_name, 'r') as f:
        
        print("\nAtrributes:")
        for key in f.attrs.keys():
            print(f"\t{key}:\t{f.attrs[key]}")
            
        print("\nGroups:")
        for key in f.keys():
            print(f"\t{key}:\tLength:\t{len(f[key][:])}")
            print(f"\tColumns: {f[key].dtype}")
            total += len(f[key][:])

        print(f"\nTotal Elements:\t{total}")
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")