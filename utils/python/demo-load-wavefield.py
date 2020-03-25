
#
# To use this script, make sure you set python path enviroment variable to include the following paths
#   ${SE2WAVE_ROOT}/utils/python:${PETSC_DIR}/lib/petsc/bin
# e.g.
#   export PYTHONPATH=${SE2WAVE_ROOT}/utils/python:${PETSC_DIR}/lib/petsc/bin
#
import numpy as np
import matplotlib.pyplot as plt

import PetscBinaryIO as pio


def se2wave_load_coordinates(filename):
  debug = True
  io = pio.PetscBinaryIO()
  data = dict()
  with open(filename) as fp:
    v = io.readInteger(fp)
    data['mx'] = v
    
    v = io.readInteger(fp)
    data['my'] = v
    
    v = io.readInteger(fp)
    data['nx'] = v
    
    v = io.readInteger(fp)
    data['ny'] = v
    
    objecttype = io.readObjectType(fp)
    v = io.readVec(fp)
    data['coor'] = v
  
  if debug:
    print('#elements-x',data['mx'])
    print('#elements-y',data['my'])
    print('#basis-x',data['nx'])
    print('#basis-y',data['ny'])

  return data

def se2wave_load_wavefield(filename,has_displacement,has_velocity):
  debug = True
  io = pio.PetscBinaryIO()
  data = dict()
  with open(filename) as fp:
    v = io.readInteger(fp)
    data['mx'] = v
    
    v = io.readInteger(fp)
    data['my'] = v
    
    v = io.readInteger(fp)
    data['nx'] = v
    
    v = io.readInteger(fp)
    data['ny'] = v

    v = io.readInteger(fp)
    data['step'] = v

    v = io.readReal(fp)
    data['time'] = v

    if has_displacement:
      objecttype = io.readObjectType(fp)
      v = io.readVec(fp)
      data['displ'] = v

    if has_velocity:
      objecttype = io.readObjectType(fp)
      v = io.readVec(fp)
      data['vel'] = v

  if debug:
    print('#elements-x',data['mx'])
    print('#elements-y',data['my'])
    print('#basis-x',data['nx'])
    print('#basis-y',data['ny'])
    print('step',data['step'])
    print('time',data['time'])
  
  return data


filename = "/Users/dmay/codes/se2wave-dev/se2wave/default_mesh_coor.pbin"
se2_coor = se2wave_load_coordinates(filename);
print(se2_coor['coor'])

w_filename = "/Users/dmay/codes/se2wave-dev/se2wave/step-1400_wavefield.pbin"
se2_field = se2wave_load_wavefield(w_filename,True,True);
print(se2_field['displ'])

