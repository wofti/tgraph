# tgraph - a python based program to plot 1D and 2D data files
# Copyright (C) 2015 Wolfgang Tichy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# directly import some math funcs from numpy into local namespace
from numpy import arccos, ceil, frexp, radians, arccosh, copysign, exp, log, \
  sin, arcsin, cos, expm1, log10, sinh, arcsinh, cosh, fabs, log1p, sqrt, \
  arctan, degrees, isinf, modf, tan, arctan2, e, floor, \
  isnan, pi, tanh, arctanh, fmod, ldexp, power, trunc
import numpy as np
import struct

# only needed to check python version
import sys

################################################################
# return 2nd order diff of 1d numpy array
# we can get dy/dx from D(y)/D(x)
def D(x):
  N = len(x)
  dx = np.zeros(N)
  if N<2:
    return dx
  for i in range(1, N-1):
    dx[i] = (x[i+1] - x[i-1])*0.5
  dx[0]   = x[1] - x[0]
  dx[N-1] = x[N-1] - x[N-2]
  return dx

################################################################
# find listindex closest to t
def geti_closest_to_t(timelist, t):
  dtm = 1.1*abs(float(timelist[0]) - float(t))
  i = 0
  for time in timelist:
    dt = abs(float(time) - float(t))
    if dt<dtm:
      dtm = dt
      im = i
    i += 1
  return im

################################################################
# find listindex at or closest to t
def geti_from_t(timelist, t, return_closest_t=1, tol=0.5):
  ni = len(timelist)
  # try to find i by direct comparison
  i = 0
  for time in timelist:
    if time == t:
      im = i
      break
    i += 1
  # if we found the exact time return it
  if i<ni:
    return im
  # otherwise we compute i with closest time or return a negative i
  icl = geti_closest_to_t(timelist, t)
  if return_closest_t == 1:
    return icl
  else:
    if icl<ni-1:
      ip = icl+1
    else:
      ip = icl
    if icl>0:
      im = icl-1
    else:
      im = icl
    # get time step and error in time if we use time at i=icl
    dt = min(abs(timelist[ip] - timelist[icl]),
             abs(timelist[im] - timelist[icl]))
    err = abs(timelist[icl] - t)
    # compare err with dt
    if err > dt * tol:
      return -1
    else:
      return icl


################################################################
# find and get value of a parameter
def getparameter(line, par):
  val = ''
  ok = 0
  EQsign = 0
  p = line.find(par)
  # make sure there is no letter in front the par name we find
  if p > 0:
    before = line[p-1:p]
    if before.isalpha():
      p=-1
  if p >= 0:
    ok = 1
    afterpar = line[p+len(par):]
    # strip white space at end and beginning
    afterpar  = afterpar.rstrip()
    afterpar2 = afterpar.lstrip()
    if len(afterpar2) > 0:
      # if '=' is there
      if afterpar2[0] == '=':
        afterpar2 = afterpar2[1:]
        val = afterpar2.lstrip()
        EQsign = 1
      # if we have a space instead
      elif afterpar[0].isspace():
        val = afterpar.lstrip()
        EQsign = 0

  return (val , ok, EQsign)


################################################################
# is line data or comment or time = ...
def linetype(line, timestr='time'):
  iscomment = 0
  foundtime = 0
  time = ''
  # look for comments
  lstart = line.lstrip()
  if len(lstart) == 0:
    lstart = line
  if lstart[0] == '#' or lstart[0] == '"' or lstart[0] == '%':
    iscomment = 1
  # look for time value
  (time, ok, EQ) = getparameter(line.lower(), timestr)
  if len(time) > 0:
    time = time.split()[0]
  # see if we found time
  # if ok==1 and EQ==1:
  if ok==1:
    foundtime = 1
    # look for junk and cut it out
    l = len(time)
    p1 = time.find('"')
    p2 = time.find(',')
    if p1 < 0:
      p1 = l
    if p2 < 0:
      p2 = l
    pm = min(p1, p2)
    time = time[:pm]
  else:
    fondtime = 0

  return (iscomment, foundtime, time)

################################################################
# convert a string to float similar to C's atof
def WT_atof(str, strfl=0.0):
  fl = float(strfl)
  if len(str) == 0:
    return fl
  list = str.split()
  str1 = list[0]
  while len(str1)>0:
    try:
      fl = float(str1)
      break
    except:
      str1 = str1[:-1]
  return fl

################################################################
# replace all inf in array x by 1e300
def inf_in_numpyarray_to_1e300(x):
  maxx = np.nanmax(x)
  minx = np.nanmin(x)
  if maxx==np.inf:
    x[x==+np.inf] = +1e300
  if minx==-np.inf:
    x[x==-np.inf] = -1e300
  return x

################################################################
# replace inf by 1e300 in number x
def inf_to_1e300(x):
  if x==np.inf:
    return 1e300
  elif x==-np.inf:
    return -1e300
  else:
    return x

################################################################
# pad jagged 2D list Ls
def pad_jagged_2D_list(Ls, colsmin=1, padval=0.0):
  rows = len(Ls)
  maxrl = colsmin
  pads = 0
  # find max row length maxrl
  for i in range(rows):
    rl = len(Ls[i])
    if rl>maxrl:
      maxrl = rl
  # pad all rows
  for i in range(rows):
    rl = len(Ls[i])
    for j in range(maxrl-rl):
      Ls[i].append(padval)
      pads += 1
  return pads


################################################################
# functions to read simple VTK files

# find out what kind of vtk file we have
def determine_vtk_DATASET_type(filename):
  with open(filename, 'rb') as f:
    # go over lines until DATASET
    while True:
      line = f.readline()
      if not line:
        break
      (val,ok,EQsign) = getparameter(line.decode('ascii'), 'DATASET')
      if ok == 1:
        DATASET = val
        break
    return DATASET


# load data from bam vtk file for Cartesian grids
def load_vtk_STRUCTURED_POINTS_data(filename, timestr):
  with open(filename, 'rb') as f:
    varname = ''
    time = '0'
    BINARY = 0
    DATASET = ''
    nx = 1
    ny = 1
    nz = 1
    x0 = 0.0
    y0 = 0.0
    z0 = 0.0
    dx = 1.0
    dy = 1.0
    dz = 1.0
    double_prec = 0
    # go over lines until LOOKUP_TABLE
    while True:
      line = f.readline()
      if not line:
        break
      (val,ok,EQsign) = getparameter(line.lower().decode('ascii'), 'variable')
      if ok == 1:
        varname = val
        p = val.find(',')
        if p >= 1:
          varname = val[:p-1]
      (val,ok,EQsign) = getparameter(line.lower().decode('ascii'), timestr)
      if ok == 1:
        time = val
        p = val.find(',')
        if p >= 1:
          time = val[:p-1]
      (val,ok,EQsign) = getparameter(line.decode('ascii'), 'BINARY')
      if ok == 1:
        BINARY = 1
      (val,ok,EQsign) = getparameter(line.decode('ascii'), 'DATASET')
      if ok == 1:
        DATASET = val
      (val,ok,EQsign) = getparameter(line.decode('ascii'), 'DIMENSIONS')
      if ok == 1:
        slist = val.split()
        nx = int(slist[0])
        ny = int(slist[1])
        nz = int(slist[2])
      (val,ok,EQsign) = getparameter(line.decode('ascii'), 'ORIGIN')
      if ok == 1:
        slist = val.split()
        x0 = float(slist[0])
        y0 = float(slist[1])
        z0 = float(slist[2])
      (val,ok,EQsign) = getparameter(line.decode('ascii'), 'SPACING')
      if ok == 1:
        slist = val.split()
        dx = float(slist[0])
        dy = float(slist[1])
        dz = float(slist[2])
      (val,ok,EQsign) = getparameter(line.decode('ascii'), 'SCALARS')
      if ok == 1:
        p = val.find('double')
        if p >= 0:
          double_prec = 1
      p = line.decode('ascii').find('LOOKUP_TABLE')
      if p >= 0:
        break
    # once we get here, we have read the ASCII header and now the data start
    npoints = nx*ny*nz
    if BINARY == 1:
      vdata = read_raw_binary_vtk(f, npoints, double_prec)
    else:
      vdata = read_raw_text_vtk(f, npoints)
    # now make x,y,z coords for all points
    # xr = np.linspace(x0, x0 + dx*(nx-1), nx)
    # yr = np.linspace(y0, y0 + dy*(ny-1), ny)
    # zr = np.linspace(z0, z0 + dz*(nz-1), nz)
    # first make empty numpy array of the correct type
    if double_prec == 1:
      xyzdata = np.empty(3*npoints)
    else:
      xyzdata = np.empty(3*npoints, dtype=np.float32)
    # now insert coords
    ijk = 0
    for k in range(0,nz):
      z = z0 + k*dz
      for j in range(0,ny):
        y = y0 + j*dy
        for i in range(0,nx):
          xyzdata[ijk]   = x0 + i*dx
          xyzdata[ijk+1] = y
          xyzdata[ijk+2] = z
          ijk += 3
    xyzdata = xyzdata.reshape(-1,3)
    # figure out blocking for mesh grid
    if nz == 1:
      blocks = ny
    elif ny == 1 or nx == 1:
      blocks = nz
    else:
      blocks = nz
    data = np.concatenate((xyzdata , vdata.reshape(-1,1)), axis=1)
    return (data, WT_atof(time), blocks)


# load data from sgrid vtk file for grids with non-uniform spacing
def load_vtk_RECTILINEAR_GRID_data(filename, timestr):
  with open(filename, 'rb') as f:
    varname = ''
    time = '0'
    BINARY = 0
    DATASET = ''
    nx = 1
    ny = 1
    nz = 1
    double_prec = 0
    # go over lines until LOOKUP_TABLE
    while True:
      line = f.readline()
      if not line:
        break
      (val,ok,EQsign) = getparameter(line.lower().decode('ascii'), 'variable')
      if ok == 1:
        varname = val
        p = val.find(',')
        if p >= 1:
          varname = val[:p-1]
      (val,ok,EQsign) = getparameter(line.lower().decode('ascii'), timestr)
      if ok == 1:
        time = val
        p = val.find(',')
        if p >= 1:
          time = val[:p-1]
      (val,ok,EQsign) = getparameter(line.decode('ascii'), 'BINARY')
      if ok == 1:
        BINARY = 1
      (val,ok,EQsign) = getparameter(line.decode('ascii'), 'DATASET')
      if ok == 1:
        DATASET = val
      (val,ok,EQsign) = getparameter(line.decode('ascii'), 'DIMENSIONS')
      if ok == 1:
        slist = val.split()
        nx = int(slist[0])
        ny = int(slist[1])
        nz = int(slist[2])
      (val,ok,EQsign) = getparameter(line.decode('ascii'), 'X_COORDINATES')
      if ok == 1:
        slist = val.split()
        nX = int(slist[0])
        p = val.find('double')
        if p >= 0:
          double_prec = 1
        if BINARY == 1:
          Xdata = read_raw_binary_vtk(f, nX, double_prec)
        else:
          Xdata = read_raw_text_vtk(f, nX)
      (val,ok,EQsign) = getparameter(line.decode('ascii'), 'Y_COORDINATES')
      if ok == 1:
        slist = val.split()
        nY = int(slist[0])
        p = val.find('double')
        if p >= 0:
          double_prec = 1
        if BINARY == 1:
          Ydata = read_raw_binary_vtk(f, nY, double_prec)
        else:
          Ydata = read_raw_text_vtk(f, nY)
      (val,ok,EQsign) = getparameter(line.decode('ascii'), 'Z_COORDINATES')
      if ok == 1:
        slist = val.split()
        nZ = int(slist[0])
        p = val.find('double')
        if p >= 0:
          double_prec = 1
        if BINARY == 1:
          Zdata = read_raw_binary_vtk(f, nZ, double_prec)
        else:
          Zdata = read_raw_text_vtk(f, nZ)
      (val,ok,EQsign) = getparameter(line.decode('ascii'), 'SCALARS')
      if ok == 1:
        p = val.find('double')
        if p >= 0:
          double_prec = 1
      p = line.decode('ascii').find('LOOKUP_TABLE')
      if p >= 0:
        break
    # once we get here, we have read the ASCII header and now the data start
    npoints = nx*ny*nz
    if BINARY == 1:
      vdata = read_raw_binary_vtk(f, npoints, double_prec)
    else:
      vdata = read_raw_text_vtk(f, npoints)
    # now make x,y,z coords for all points
    # first make empty numpy array of the correct type
    if double_prec == 1:
      xyzdata = np.empty(3*npoints)
    else:
      xyzdata = np.empty(3*npoints, dtype=np.float32)
    # now insert coords
    ijk = 0
    for k in range(0,nZ):
      for j in range(0,nY):
        for i in range(0,nX):
          xyzdata[ijk]   = Xdata[i]
          xyzdata[ijk+1] = Ydata[j]
          xyzdata[ijk+2] = Zdata[k]
          ijk += 3
    xyzdata = xyzdata.reshape(-1,3)
    # figure out blocking for mesh grid
    if nz == 1:
      blocks = ny
    elif ny == 1 or nx == 1:
      blocks = nz
    else:
      blocks = nz
    data = np.concatenate((xyzdata , vdata.reshape(-1,1)), axis=1)
    return (data, WT_atof(time), blocks)


# load data from bam vtk file for STRUCTURED_GRID (e.g. polar) grids
def load_vtk_STRUCTURED_GRID_data(filename, timestr):
  with open(filename, 'rb') as f:
    varname = ''
    time = '0'
    BINARY = 0
    DATASET = ''
    nx = 1
    ny = 1
    nz = 1
    double_prec = 0
    nPOINTS = 0

    # go over lines until POINTS
    while True:
      line = f.readline()
      if not line:
        break
      (val,ok,EQsign) = getparameter(line.lower().decode('ascii'), 'variable')
      if ok == 1:
        varname = val
        p = val.find(',')
        if p >= 1:
          varname = val[:p-1]
      (val,ok,EQsign) = getparameter(line.lower().decode('ascii'), timestr)
      if ok == 1:
        time = val
        p = val.find(',')
        if p >= 1:
          time = val[:p-1]
      (val,ok,EQsign) = getparameter(line.decode('ascii'), 'BINARY')
      if ok == 1:
        BINARY = 1
      (val,ok,EQsign) = getparameter(line.decode('ascii'), 'DATASET')
      if ok == 1:
        DATASET = val
      (val,ok,EQsign) = getparameter(line.decode('ascii'), 'DIMENSIONS')
      if ok == 1:
        slist = val.split()
        nx = int(slist[0])
        ny = int(slist[1])
        nz = int(slist[2])
      (val,ok,EQsign) = getparameter(line.decode('ascii'), 'POINTS')
      if ok == 1:
        nPOINTS = int( WT_atof(val) )
        p = val.find('double')
        if p >= 0:
          double_prec = 1
      p = line.decode('ascii').find('POINTS')
      if p >= 0:
        break
    # once we get here, we have read the ASCII header for the points
    # and now the data for the grid points start
    npoints = nx*ny*nz
    if(nPOINTS>0):
      npoints = nPOINTS
    ncoords = npoints * 3   # there x,y,z coords for each point
    if BINARY == 1:
      vdata = read_raw_binary_vtk(f, ncoords, double_prec)
    else:
      vdata = read_raw_text_vtk(f, ncoords)
    # now read x,y,z coords for all points from vdata
    # e.g. xyzdata[11,0] is x-coord of point 11
    xyzdata = vdata.reshape(-1,3)
    # figure out blocking for mesh grid
    if nz == 1:
      blocks = ny
    elif ny == 1 or nx == 1:
      blocks = nz
    else:
      blocks = nz

    # go over data text lines until LOOKUP_TABLE
    while True:
      line = f.readline()
      if not line:
        break
      (val,ok,EQsign) = getparameter(line.decode('ascii'), 'BINARY')
      if ok == 1:
        BINARY = 1
      (val,ok,EQsign) = getparameter(line.decode('ascii'), 'ASCII')
      if ok == 1:
        BINARY = 0
      (val,ok,EQsign) = getparameter(line.decode('ascii'), 'SCALARS')
      if ok == 1:
        p = val.find('double')
        if p >= 0:
          double_prec = 1
        else:
          double_prec = 0
      p = line.decode('ascii').find('LOOKUP_TABLE')
      if p >= 0:
        break
    # once we get here, we have read the ASCII header and now the data start
    vdata = []
    if BINARY == 1:
      vdata = read_raw_binary_vtk(f, npoints, double_prec)
    else:
      vdata = read_raw_text_vtk(f, npoints)
    data = np.concatenate((xyzdata , vdata.reshape(-1,1)), axis=1)
    return (data, WT_atof(time), blocks)


################################################################
# functions to read big endian doubles or floats from binary
# or text files

# read doubles or floats from file and return in numpy array
def read_raw_binary_vtk(file, npoints, double_prec):
  if double_prec == 1:
    # read data into a byte string
    bstr = file.read(8*npoints)
    # unpack bstr into tuple of C-doubles, assuming big-endian (>) byte order
    fmt = '>%dd' % (npoints)
    dtuple = struct.unpack(fmt, bstr)
    # convert tuple dtuple into numpy array
    vdata = np.array(dtuple)
  else:
    # read data into a byte string
    bstr = file.read(4*npoints)
    # unpack bstr into tuple of C-floats, assuming big-endian (>) byte order
    fmt = '>%df' % (npoints)
    dtuple = struct.unpack(fmt, bstr)
    # convert tuple dtuple into numpy array
    vdata = np.array(dtuple, dtype=np.float32)
  return vdata

# read data from text file
def read_raw_text_vtk(file, npoints):
  dat = []
  i = 0
  while i<npoints:
    line = file.readline()
    if not line:
      print(file)
      print('read_raw_text_vtk: npoints =', npoints,
            '. But EOF already at i = ', i)
      print('appending zeros...')
      for k in range(i, npoints):
        dat.append(0.0)
      break
    # split line into a list li and append each element of li as float to dat
    li = line.split()
    for l in li:
      dat.append(float(l))
    i = i + len(li)
  vdata = np.array(dat)
  return vdata


################################################################
# a class that contains all the data in file at one time
class tTimeFrame:
  # init one timeframe
  def __init__(self, data, time=0, xcol=0, ycol=1, zcol=2, vcol=1, blocks=1):
    self.time = time
    self.xcol = xcol
    self.ycol = ycol
    self.zcol = zcol
    self.vcol = vcol
    self.data = data
    self.blocks = blocks

  # get cols with x-data and values, i.e. v-data
  def getx(self):
    return self.data[:,self.xcol]
  def gety(self):
    return self.data[:,self.ycol]
  def getz(self):
    return self.data[:,self.zcol]
  def getv(self):
    return self.data[:,self.vcol]

  # merge cols from self.data, tf2.data
  def merge_with(self, tf2):
    self.data = np.concatenate((self.data, tf2.data), axis=1)

  # add ecols empty cols to data
  def add_empty_cols(self, ecols=1):
    nrows = self.data.shape[0]
    edata = np.empty(nrows*ecols, dtype=self.data.dtype)
    newdata = np.concatenate((self.data, edata.reshape(-1,ecols)), axis=1)
    self.data = newdata

  # transform according to string in trafo, e.g. trafo = 'c[3] = 2*c[2] + t'
  def transform_col(self, trafo, c_index_shift=0):
    if len(trafo)<6:
      return
    # set time
    t = self.time
    time = t
    # find col num in first arg
    s0 = trafo.split(']')[0]
    s1 = s0.split('[')[1]
    col = int(s1)
    # col index in actual data
    datacol = col - c_index_shift
    # number of cols
    ncols = self.data.shape[1]
    # add extra cols if datacol does not exist yet
    ecols = datacol - ncols + 1
    #print(self.data.shape)
    #print(datacol, ncols, ecols)
    if ecols > 0:
      self.add_empty_cols(ecols=ecols)
    #print(self.data.shape)
    ncols = self.data.shape[1]
    # put all cols in c[i]
    c = []
    for i in range(c_index_shift):
      c.append(self.data[:,0])
    for i in range(ncols):
      c.append(self.data[:,i])
    # exec code in trafo
    exec(trafo)
    # put c[col] back in data
    self.data[:,datacol] = c[col]

################################################################
# several time frames and their meta data
class tTimeFrameSet:

  # get a list that contains all times
  def get_timelist(self):
    t = []
    for ttf in self.timeframes:
      t.append(ttf.time)
    return t   

  def __init__(self, filename, timestr):
    # make tTimeFrame for each time in file
    self.timeframes = []  # init timeframes
    # get extension from filename
    p = filename.rfind('.')
    if p >= 0:
      ext = filename[p:]
    else:
      ext = ''
    # print(ext)
    # check filename extension
    if ext.find('.vtk') >= 0:
      # find DATASET type
      DATASET = determine_vtk_DATASET_type(filename)
      # print(DATASET)
      if DATASET.find('STRUCTURED_GRID')>=0:
        (dat, time, bl) = load_vtk_STRUCTURED_GRID_data(filename, timestr)
      elif DATASET.find('RECTILINEAR_GRID')>=0:
        (dat, time, bl) = load_vtk_RECTILINEAR_GRID_data(filename, timestr)
      else:
        (dat, time, bl) = load_vtk_STRUCTURED_POINTS_data(filename, timestr)
      self.timeframes.append(tTimeFrame(dat, time, blocks=bl))
      # make a list with times from all timeframes
      self.timelist = self.get_timelist()
    else: # assume text files are used
      # print('TEXT file!')
      self.append_textfile_data(filename, timestr)

  # read the data from a text file and append to self.timeframes
  def append_textfile_data(self, filename, timestr):
    # check python version
    if sys.version_info[0] < 3:
      open_noerr = lambda fname,mode: open(fname,mode)
    else:
      open_noerr = lambda fname,mode: open(fname,mode, errors='ignore')
    # open file and ignore encoding errors if we have python3
    with open_noerr(filename, 'r') as f:
      dat = []
      time = 0
      time_n = -1e+300 # new time found in file
      tnum = 1 # time number
      nl_num = 0
      prev_was_nl = 0
      lnum = 1
      for line in f:
        # look for time
        (iscomment, foundtime, time0) = linetype(line, timestr)
        if foundtime == 1:
          # if we found time but dat is not empty write. We are at end
          # of timeframe.
          if dat != []:
            if prev_was_nl == 0:
              nl_num += 1
            pad_jagged_2D_list(dat)
            dat = np.array(dat)
            self.timeframes.append(tTimeFrame(dat, time, blocks=nl_num))
            dat = []
          # reset newline and line counters
          nl_num = 0
          prev_was_nl = 0
          lnum = 1
          # save old time and then get new time
          time_o = time_n
          time_n = WT_atof(time0, strfl=tnum)
          # if time_o == time_n increase time by a little bit
          if time_o == time_n:
            if time == 0.0:
              time = 1e-300
            time = 1.00000000000001 * time
          else: # if time_o != time_n, set time to new time
            time = time_n
          tnum += 1
        elif iscomment == 1:
          pass
        elif line.isspace():  # e.g. if line == '\n':
          # count '\n' after some data, but omit duplicates
          if prev_was_nl == 0 and dat != []:
            nl_num += 1
          prev_was_nl = 1
        else: # there is no time, so now we have a data row
          row = line.split() # make list from line
          row = [WT_atof(n, strfl=lnum) for n in row] # convert list to float
          dat.append(row)
          prev_was_nl = 0
        lnum += 1
      # arrived at end of file, 
      # so write last piece of data, if it's not empty:
      if dat != []:
        pad_jagged_2D_list(dat)
        dat = np.array(dat)
        if prev_was_nl == 0:
          nl_num += 1
        self.timeframes.append(tTimeFrame(dat, time, blocks=nl_num))
    # make a list with times from all timeframes
    self.timelist = self.get_timelist()

  # set values that say where data columns are
  def set_cols(self, xcol=0, ycol=1, zcol=2, vcol=1):
    for ttf in self.timeframes:
      ttf.xcol = xcol
      ttf.ycol = ycol
      ttf.zcol = zcol
      ttf.vcol = vcol

  # set one of the values that say where data columns are
  def set_xcols(self, col):
    for ttf in self.timeframes: ttf.xcol = col
  def set_ycols(self, col):
    for ttf in self.timeframes: ttf.ycol = col
  def set_zcols(self, col):
    for ttf in self.timeframes: ttf.zcol = col
  def set_vcols(self, col):
    for ttf in self.timeframes: ttf.vcol = col

  # return the values that say where data columns are
  def get_xcol0(self):
    return self.timeframes[0].xcol
  def get_ycol0(self):
    return self.timeframes[0].ycol
  def get_zcol0(self):
    return self.timeframes[0].zcol
  def get_vcol0(self):
    return self.timeframes[0].vcol

  def gettime_i(self, i):
    return self.timeframes[i].time
  def getx_i(self, i):
    return self.timeframes[i].getx()
  def gety_i(self, i):
    return self.timeframes[i].gety()
  def getz_i(self, i):
    return self.timeframes[i].getz()
  def getv_i(self, i):
    return self.timeframes[i].getv()
  def getblocks_i(self, i):
    return self.timeframes[i].blocks

  # return min and max of time
  def mintime(self):
    return min(self.timelist)
  def maxtime(self):
    return max(self.timelist)

  # get time index i closest to t
  def geti_t(self, t, retcl=1):
    return geti_from_t(self.timelist, t, retcl)

  # get x,y,z, v from time
  def getx(self, time, retcl=1):
    i = self.geti_t(time, retcl)
    if(i>=0):
      return self.getx_i(i)
    else:
      return []
  def gety(self, time, retcl=1):
    i = self.geti_t(time, retcl)
    if(i>=0):
      return self.gety_i(i)
    else:
      return []
  def getz(self, time, retcl=1):
    i = self.geti_t(time, retcl)
    if(i>=0):
      return self.getz_i(i)
    else:
      return []
  def getv(self, time, retcl=1):
    i = self.geti_t(time, retcl)
    if(i>=0):
      return self.getv_i(i)
    else:
      return []
  def getblocks(self, time, retcl=1):
    i = self.geti_t(time, retcl)
    if(i>=0):
      return self.getblocks_i(i)
    else:
      return []

  # get min and max of x
  def minx(self):
    m0 = np.nanmin(self.getx_i(0))
    for i in range(1, len(self.timelist)):
      m = np.nanmin(self.getx_i(i))
      if m<m0:
        m0 = m
    return m0
  def maxx(self):
    m0 = np.nanmax(self.getx_i(0))
    for i in range(1, len(self.timelist)):
      m = np.nanmax(self.getx_i(i))
      if m>m0:
        m0 = m
    return m0

  # get min and max of y
  def miny(self):
    m0 = np.nanmin(self.gety_i(0))
    for i in range(1, len(self.timelist)):
      m = np.nanmin(self.gety_i(i))
      if m<m0:
        m0 = m
    return m0
  def maxy(self):
    m0 = np.nanmax(self.gety_i(0))
    for i in range(1, len(self.timelist)):
      m = np.nanmax(self.gety_i(i))
      if m>m0:
        m0 = m
    return m0

  # get min and max of z
  def minz(self):
    m0 = np.nanmin(self.getz_i(0))
    for i in range(1, len(self.timelist)):
      m = np.nanmin(self.getz_i(i))
      if m<m0:
        m0 = m
    return m0
  def maxz(self):
    m0 = np.nanmax(self.getz_i(0))
    for i in range(1, len(self.timelist)):
      m = np.nanmax(self.getz_i(i))
      if m>m0:
        m0 = m
    return m0

  # get min and max of v
  def minv(self):
    m0 = np.nanmin(self.getv_i(0))
    for i in range(1, len(self.timelist)):
      m = np.nanmin(self.getv_i(i))
      if m<m0:
        m0 = m
    return m0
  def maxv(self):
    m0 = np.nanmax(self.getv_i(0))
    for i in range(1, len(self.timelist)):
      m = np.nanmax(self.getv_i(i))
      if m>m0:
        m0 = m
    return m0

  # merge with other timeframeset
  def merge_with(self, tfs2):
    i=0
    for tf in self.timeframes:
      tf.merge_with(tfs2.timeframes[i])
      i += 1

  # append other timeframeset
  def append_other(self, tfs2):
    for tf in tfs2.timeframes:
      self.timeframes.append(tf)
    # make a list with times from all timeframes
    self.timelist = self.get_timelist()

  # add ecols empty cols to the entire set
  def add_empty_cols(self, ecols=1):
    for ttf in self.timeframes: 
      ttf.add_empty_cols(ecols=ecols)

  # transform col in entire set
  def transform_col(self, trafo, c_index_shift=0):
    for ttf in self.timeframes: 
      ttf.transform_col(trafo, c_index_shift=c_index_shift)

################################################################
# a structure to hold file names and their data
class FileData:
  pass


################################################################
# a list of files and their data
class tFileList:
  # init list of files
  def __init__(self):
    self.file = []      # init file list

  # add a file to list
  def add(self, filename, timestr='time'):
    with open(filename, 'r') as f:
      filedata = FileData()
      filedata.filename = filename
      filedata.name = filename
      filedata.data = tTimeFrameSet(filename, timestr)
      self.file.append(filedata)

  # remove a file at index i
  def remove(self, i):
    del self.file[i]

  # merge 2 files in list
  def merge_file_i2_into_i1(self, i1, i2):
    dat1 = self.file[i1].data
    dat2 = self.file[i2].data
    fn1 = self.file[i1].filename
    fn2 = self.file[i2].filename
    n1 = self.file[i1].name
    n2 = self.file[i2].name
    self.file[i1].filename = fn2
    self.file[i1].name = n1 + ' ' + n2
    dat1.merge_with(dat2)
    del self.file[i2]

  # append file i2 to i1 in list
  def append_file_i2_to_i1(self, i1, i2):
    dat1 = self.file[i1].data
    dat2 = self.file[i2].data
    dat1.append_other(dat2)
    del self.file[i2]

  def getdataset_i(self, i):
    return self.file[i].data

  # make list with all times from all data sets
  def get_timelist(self):
    t = []
    for f in self.file:
      dst = f.data.timelist
      for time in dst:
        t.append(time)
    T = set(t)   # remove duplicates
    t = list(T)
    t.sort()
    return t

  # find min and max time in all data
  def mintime(self):
    t = []
    for f in self.file:
      t.append(f.data.mintime())
    return min(t)
  def maxtime(self):
    t = []
    for f in self.file:
      t.append(f.data.maxtime())
    return max(t)

  # find min and max x in all data
  def minx(self):
    m = []
    for f in self.file:
      m.append(f.data.minx())
    return min(m)
  def maxx(self):
    m = []
    for f in self.file:
      m.append(f.data.maxx())
    return max(m)

  # find min and max y in all data
  def miny(self):
    m = []
    for f in self.file:
      m.append(f.data.miny())
    return min(m)
  def maxy(self):
    m = []
    for f in self.file:
      m.append(f.data.maxy())
    return max(m)

  # find min and max z in all data
  def minz(self):
    m = []
    for f in self.file:
      m.append(f.data.minz())
    return min(m)
  def maxz(self):
    m = []
    for f in self.file:
      m.append(f.data.maxz())
    return max(m)
    
  # find min and max v in all data
  def minv(self):
    m = []
    for f in self.file:
      m.append(f.data.minv())
    return min(m)
  def maxv(self):
    m = []
    for f in self.file:
      m.append(f.data.maxv())
    return max(m)



## test
##s = tTimeFrameSet('/home/wolf/wolfGIT/MyPapers/IsometricEmbedding/plots/IE_r_dentedSphere/IE_K.Y0')      
#s = tTimeFrameSet('l')
#print('s=', s)
#
##print(s.timeframes[0].data)
#
##print(s.timeframes[0].getx())
#s.set_cols(0,1)
##print(s.timeframes[0].getx())
#
#for t in s.timeframes:
#  print(t.time, t.data, t.getx(), t.getv())
