# Imports
import sys
import os
sys.path.insert(0,'/global/cfs/projectdirs/lsst/groups/SRV/gcr-catalogs')
# reader with parallel read

from DW_to_VM_map import DimrothWatsonToVonMisesMapper

from modular_alignments.modular_alignment import project_alignments_with_NCP, get_position_angle, phi_to_e1_e2
from modular_alignments.modular_alignment_2d import tidal_angle
from modular_alignments.modular_alignment import align_to_axis as align_to_axis_3d
from modular_alignments.modular_alignment_2d import align_to_axis as align_to_axis_2d

# Get MPI stuff
from mpi4py import MPI

import GCRCatalogs as gcrc
from astropy.table import Table
from astropy.io import ascii
import numpy as np