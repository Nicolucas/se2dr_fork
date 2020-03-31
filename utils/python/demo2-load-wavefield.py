
#
# To use this script, make sure you set python path enviroment variable to include the following paths
#   ${SE2WAVE_ROOT}/utils/python:${PETSC_DIR}/lib/petsc/bin
# e.g.
#   export PYTHONPATH=${SE2WAVE_ROOT}/utils/python:${PETSC_DIR}/lib/petsc/bin
#

import os as os
import se2waveload as load

path = "/Users/dmay/codes/se2wave-dev/se2wave/"

m_filename = os.path.join(path,"step-0100_wavefield.json")
se2 = load.se2wave_load_json(m_filename,debug=True)
print('wavefield filename ->',se2['se2wave']['data']['filename'])
has_displacement = False
has_velocity = False
fields = se2['se2wave']['data']['fields']
if 'u' in fields:
  has_displacement = True
if 'v' in fields:
  has_velocity = True
print('fields:',se2['se2wave']['data']['fields'])
print('fields: displacement?',has_displacement)
print('fields: vel?',has_velocity)

heavydata_filename = os.path.join(path,se2['se2wave']['data']['filename'])
se2_field = load.se2wave_load_wavefield(heavydata_filename,has_displacement,has_velocity,debug=True);

if has_displacement:
  mm = min(abs(se2_field['displ']))
  print('min |u|',mm)
  mm = max(abs(se2_field['displ']))
  print('max |u|',mm)
if has_velocity:
  mm = min(abs(se2_field['vel']))
  print('min |v|',mm)
  mm = max(abs(se2_field['vel']))
  print('max |v|',mm)

m_filename = os.path.join(path,"default_mesh_coor.json")
se2 = load.se2wave_load_json(m_filename,debug=True)
print('coor filename ->',se2['se2wave']['data']['filename'])

heavydata_filename = os.path.join(path,se2['se2wave']['data']['filename'])
se2_coor = load.se2wave_load_coordinates(heavydata_filename,debug=True);
print(se2_coor['coor'])





