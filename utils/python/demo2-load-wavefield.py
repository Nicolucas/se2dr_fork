
#
# To use this script, make sure you set python path enviroment variable to include the following paths
#   ${SE2WAVE_ROOT}/utils/python:${PETSC_DIR}/lib/petsc/bin
# e.g.
#   export PYTHONPATH=${SE2WAVE_ROOT}/utils/python:${PETSC_DIR}/lib/petsc/bin
#

import PetscBinaryIO as pio


def se2wave_load_coordinates(filename):
  debug = False
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
    print('<debug> se2wave_load_coordinates()')
    print('<debug> #elements-x',data['mx'])
    print('<debug> #elements-y',data['my'])
    print('<debug> #basis-x',data['nx'])
    print('<debug> #basis-y',data['ny'])

  return data

def se2wave_load_wavefield(filename,has_displacement,has_velocity):
  debug = False
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
    print('<debug> se2wave_load_wavefield()')
    print('<debug> #elements-x',data['mx'])
    print('<debug> #elements-y',data['my'])
    print('<debug> #basis-x',data['nx'])
    print('<debug> #basis-y',data['ny'])
    print('<debug> step',data['step'])
    print('<debug> time',data['time'])
  
  return data


def se2wave_load_json(filename):
  import json as j
  
  debug = True
  jdata = dict()
  with open(filename, "r") as fp:
    jdata = j.load(fp)

    if debug:
      wdata = jdata['se2wave']
      print('<debug> se2wave_load_json()')
      print('<debug> spatial_dimension:',wdata['spatial_dimension'])
      print('<debug> nx:',wdata['nx'])
      print('<debug> time:',wdata['time'])
      print('<debug> fields:',wdata['data']['fields'])
      print('<debug> datafilename:',wdata['data']['filename'])
  return jdata


path = "/Users/dmay/codes/se2wave-dev/se2wave/"

m_filename = path + "step-0100_wavefield.json"
se2 = se2wave_load_json(m_filename)
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

se2_field = se2wave_load_wavefield(se2['se2wave']['data']['filename'],has_displacement,has_velocity);

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

m_filename = path + "default_mesh_coor.json"
se2 = se2wave_load_json(m_filename)
print('coor filename ->',se2['se2wave']['data']['filename'])

se2_coor = se2wave_load_coordinates(se2['se2wave']['data']['filename']);
print(se2_coor['coor'])





