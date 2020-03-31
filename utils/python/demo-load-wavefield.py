
#
# To use this script, make sure you set python path enviroment variable to include the following paths
#   ${SE2WAVE_ROOT}/utils/python:${PETSC_DIR}/lib/petsc/bin
# e.g.
#   export PYTHONPATH=${SE2WAVE_ROOT}/utils/python:${PETSC_DIR}/lib/petsc/bin
#

import se2waveload as load

filename = "/Users/dmay/codes/se2wave-dev/se2wave/default_mesh_coor.pbin"
se2_coor = load.se2wave_load_coordinates(filename);
print(se2_coor['coor'])

w_filename = "/Users/dmay/codes/se2wave-dev/se2wave/step-1400_wavefield.pbin"
se2_field = load.se2wave_load_wavefield(w_filename,True,True);
print(se2_field['displ'])

