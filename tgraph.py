#!/usr/bin/env python

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

import sys

# check python version
if sys.version_info[0] < 3:
  from Tkinter import *
  if sys.version_info[1] > 6:
    from ttk import *        # overrides some tkinter stuff
  import tkFileDialog as filedialog
else:
  from tkinter import *
  from tkinter.ttk import *  # overrides some tkinter stuff
  import tkinter.filedialog as filedialog

# for 2d
import matplotlib as mpl
# for 3d
from mpl_toolkits.mplot3d import  axes3d,Axes3D
from matplotlib import cm
# for tkinter
mpl.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
try:
  from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavToolbar
except:
  from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg as NavToolbar
# since "import matplotlib as mpl" does not import figure we get it explicitly
from matplotlib.figure import Figure

import numpy as np

# my data classes
import tdata

######################################################################
# tgraph version number
tgraph_version = "1.9"
print('tgraph', tgraph_version)

# get dir where tdata.py is found, and construct filename of tgraph.txt
import os
tdata_dir = os.path.dirname(tdata.__file__)
tgraph_txt_file = os.path.join(tdata_dir, 'tgraph.txt')

######################################################################
# default cols:
xcol = 0
ycol = 1
zcol = 2
vcol = 1
#print('Default cols:', xcol+1,ycol+1,zcol+1,':', vcol+1)
print('Default cols:', xcol+1,ycol+1,':', vcol+1)

# default stride
graph_stride = 1

######################################################################
# load data from command line arguments
filelist = tdata.tFileList()
argvs = sys.argv[1:]
print('Trying to open files:')
gotC = 0
gotx = 0
goty = 0
gotv = 0
gots = 0
gott = 0
timelabel_str = 'time'
got_xrange = 0
got_yrange = 0
got_vrange = 0
openCBrack = 0
inCBrack = 0
openSBrack = 0
inSBrack = 0
endSBrack = 0
for argv in argvs:
  # check for new -c opt
  pos = argv.find('-c')
  if pos == 0:
    gotC = 1
    continue
  # check for new -x opt
  pos = argv.find('-x')
  if pos == 0:
    gotx = 1
    continue
  # check for new -y opt
  pos = argv.find('-y')
  if pos == 0:
    goty = 1
    continue
  # check for new -v opt
  pos = argv.find('-v')
  if pos == 0:
    gotv = 1
    continue
  # check for new -s opt
  pos = argv.find('-s')
  if pos == 0:
    gots = 1
    continue
  # check for new -t opt
  pos = argv.find('-t')
  if pos == 0:
    gott = 1
    continue
  # check for -m opt
  pos = argv.find('-m')
  if pos == 0:
    mpl.rcParams['lines.marker'] = 'o'
    continue
  # did we get -c opt?
  if gotC == 1:
    cols = argv.split(':')
    xcol = int(cols[0])-1
    if len(cols)==2:
      vcol = int(cols[1])-1
    if len(cols)==3:
      ycol = int(cols[1])-1
      vcol = int(cols[2])-1
    # print('cols:', xcol+1,ycol+1,zcol+1,':', vcol+1)
    print('cols:', xcol+1,ycol+1,':', vcol+1)
    gotC = 0
    continue
  # did we get -x, -y or -v opts?
  if gotx == 1 or goty == 1 or gotv == 1:
    vrange = argv.split(':')
    #print(vrange)
    if gotx == 1:
      graph_xmin = float(vrange[0])
      graph_xmax = float(vrange[1])
      got_xrange = 1
    if goty == 1:
      graph_ymin = float(vrange[0])
      graph_ymax = float(vrange[1])
      got_yrange = 1
    if gotv == 1:
      graph_vmin = float(vrange[0])
      graph_vmax = float(vrange[1])
      got_vrange = 1
    gotx = 0
    goty = 0
    gotv = 0
    continue
  # did we get -s opts?
  if gots == 1:
    graph_stride = int(argv)
    gots = 0
    continue
  # did we get -t opts?
  if gott == 1:
    timelabel_str = str(argv).lower()
    gott = 0
    continue
  # check for brackets, is there a '{' or a '}'
  pos = argv.find('{')
  if pos == 0:
    openCBrack = 1
    inCBrack = 0
    continue
  pos = argv.find('}')
  if pos == 0:
    openCBrack = 0
    inCBrack = 0
    continue
  # check for brackets, is there a '[' or a ']'
  pos = argv.find('[')
  if pos == 0:
    openSBrack = 1
    inSBrack = 0
    endSBrack = 0
    continue
  pos = argv.find(']')
  if pos == 0:
    openSBrack = 0
    inSBrack = 0
    endSBrack = 1
    # no continue here
  if endSBrack == 1:
    endSBrack = 0
  else:
    # if we get here, there was no opt, so we have a filename
    filelist.add(argv, timelabel_str)
    # print('cols:', xcol+1,ycol+1,zcol+1,':', vcol+1)
    print(filelist.file[-1].filename)
    #for tf in filelist.file[-1].data.timeframes:
    #  print('blocks =', tf.blocks)
    # set cols for the last file added
    filelist.file[-1].data.set_cols(xcol=xcol, ycol=ycol, zcol=2, vcol=vcol)
  # are we in a [ ] block so that we have to append a file?
  if inSBrack == 1:
    filelist.append_file_i2_to_i1(-2, -1)
    continue
  if openSBrack == 1:
    inSBrack = 1
    openSBrack = 0
    continue
  # are we in a { } block so that we have to merge files?
  if inCBrack == 1:
    # print(filelist.file)
    filelist.merge_file_i2_into_i1(-2, -1)
  if openCBrack == 1:
    inCBrack = 1
    openCBrack = 0

#for f in filelist.file:
#  print("timeframes of", f.name, "after merge")
#  for tf in f.data.timeframes:
#    print(tf.time)
#    print(tf.data)

if len(filelist.file) == 0:
  print('no files given on command line\n')
  print('Purpose of tgraph.py:')
  print('We can show and animate data from files that contain multiple timeframes.')
  print('Each timeframe consists of a timelabel and a number of data columns. E.g.:')
  print('# time = 1.0')
  print('1   2')
  print('2   5')
  print('3  10\n')
  print('Usage:')
  print('tgraph.py [-c 1:2[:3]] File1 File2 ... { FileX FileF } ... [ f_t1 f_t2 ]\n')
  print('Options:')
  print('-c  specifies which columns to select')
  print('{ } one can add columns from different files by enclosing files in { }')
  print('[ ] one can add timeframes from different files by enclosing files in [ ]')
  print('-x , -y , -v  specify x-, y-, value-ranges, format is: -v vmin:vmax')
  print('-t  specifies timelabel, default is: -t time')
  print('-s  specifies stride (or step size) used to sample input data')
  print('-m  mark each point\n')
  print('Examples:')
  print('# select cols 1,2 in file1 and cols 1,4 of file2,file3 added together:')
  print('tgraph.py -c 1:2 file1 -c 1:4 { file2 file3 }')
  print('# select cols 1,2,4 from t1.vtk,t2.vtk,t3.vtk that contain one timeframe each:')
  print('tgraph.py -s 10 -c 1:2:4 [ t1.vtk t2.vtk t3.vtk ]')
  print('# select x- and value-ranges for data in file1 and mark points:')
  print('tgraph.py -x 1:5 -v 0:2 -m file1')
  # exit(1)

######################################################################
# root window for app
root = Tk()
#root.tk.call('tk', 'scaling', 2.0)  # make all tk widgets and fonts bigger
root.wm_title("tgraph")

######################################################################
# init global dictionaries

# dictionaries with labels and legend
graph_labelsOn = 0
graph_labels = {}
graph_labels['title'] = ''
graph_labels['x-axis'] = ''
graph_labels['y-axis'] = ''
graph_labels['v-axis'] = ''
graph_labels['fontsize'] = mpl.rcParams['font.size']
graph_labels['timeformat'] = '%g'
graph_legendOn = 0
graph_legend = {}
graph_legend['fontsize'] = mpl.rcParams['font.size']
graph_legend['loc']      = 'upper right'
graph_legend['ncol']     = 1
graph_legend['fancybox']     = mpl.rcParams['legend.fancybox']
graph_legend['shadow']       = mpl.rcParams['legend.shadow']
if mpl.__version__ > '1.4.2':
  graph_legend['frameon']      = mpl.rcParams['legend.frameon']
  graph_legend['framealpha']   = mpl.rcParams['legend.framealpha']
graph_legend['handlelength'] = mpl.rcParams['legend.handlelength']

# dictionary with settings for graph
graph_settings = {}
if mpl.__version__ >= '1.4.2':
  graph_settings['colormap'] = 'coolwarm'
else:
  graph_settings['colormap'] = 'jet'
graph_settings['linewidth'] = mpl.rcParams['lines.linewidth']
graph_settings['antialiased'] = 0
graph_settings['shade'] = 1
graph_settings['edgecolor'] = 'none'

# dictionary with settings for graph, where we need to setup axes after change
graph_limits = {}

# dictionaries with lines colors, styles, markers and widths
graph_linecolors = {}
graph_linestyles = {}
graph_linemarkers = {}
graph_linemarkersizes = {}
graph_linewidths = {}

# dictionaries with transformations
graph_coltrafos = {}

######################################################################
# functions needed early

# function to add file by # to global dictionaries
def set_graph_globals_for_file_i(filelist, i):
  global graph_legend
  global graph_linecolors
  global graph_linestyles
  global graph_linemarkers
  global graph_linemarkersizes
  global graph_linewidths
  global graph_coltrafos
  f = filelist.file[i]
  graph_legend['#'+str(i)] = f.name
  # can we use axes.prop_cycle ?
  if mpl.__version__ < '1.5.1': # use axes.color_cycle below Matplotlib 1.5.1
    if mpl.__version__ < '1.4.2':
      color_cycle = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    else:
      color_cycle = mpl.rcParams['axes.color_cycle']
    ncolors = len(color_cycle)
    graph_linecolors['#'+str(i)] = color_cycle[i%ncolors]
  else: # use axes.prop_cycle for all other versions
    cycle_list = list(mpl.rcParams['axes.prop_cycle'])
    ncolors = len(cycle_list)
    graph_linecolors['#'+str(i)] = cycle_list[i%ncolors]['color']
  graph_linestyles['#'+str(i)] = '-'
  marker = mpl.rcParams['lines.marker']
  markersize = mpl.rcParams['lines.markersize']
  graph_linemarkers['#'+str(i)] = marker
  graph_linemarkersizes['#'+str(i)] = markersize
  graph_linewidths['#'+str(i)] = ''
  graph_coltrafos['#'+str(i)] = ''

# specify a file graphically
def open_file():
  global filelist
  global xcol
  global ycol
  global vcol
  global timelabel_str
  global graph_timelist
  fname = filedialog.askopenfilename(title='Enter Data File Name')
  if len(fname) == 0:  # if user presses cancel fname is () or '', so exit
    return
  filelist.add(fname, timelabel_str)
  # print('cols:', xcol+1,ycol+1,zcol+1,':', vcol+1)
  i = len(filelist.file)-1
  print(filelist.file[i].filename)
  # set cols for the last file added
  filelist.file[i].data.set_cols(xcol=xcol, ycol=ycol, zcol=2, vcol=vcol)
  set_graph_globals_for_file_i(filelist, i)
  # update time min and max
  graph_timelist = filelist.get_timelist()

######################################################################

# no file was given on command line, ask for one now
if len(filelist.file) == 0:
  open_file()

if len(filelist.file) == 0:
  # print('\nNo files found!')
  exit(1)

# add all files to to global dictionaries
for i in range(0, len(filelist.file)):
  set_graph_globals_for_file_i(filelist, i)

######################################################################
# set global vars
graph_time = filelist.mintime()
graph_timelist = filelist.get_timelist()
graph_timeindex = tdata.geti_from_t(graph_timelist, graph_time)
graph_delay = 1
if got_xrange != 1:
  graph_xmin = tdata.inf_to_1e300(filelist.minx())
  graph_xmax = tdata.inf_to_1e300(filelist.maxx())
if got_yrange != 1:
  graph_ymin = tdata.inf_to_1e300(filelist.miny())
  graph_ymax = tdata.inf_to_1e300(filelist.maxy())
#if got_zrange != 1:
  #graph_zmin = tdata.inf_to_1e300(filelist.minz())
  #graph_zmax = tdata.inf_to_1e300(filelist.maxz())
if got_vrange != 1:
  graph_vmin = tdata.inf_to_1e300(filelist.minv())
  graph_vmax = tdata.inf_to_1e300(filelist.maxv())
graph_3dOn = 0
graph_axis_on = 1
graph_plot_surface = 0
graph_plot_scatter = 0
graph_clear_on_replot = 1
graph_plot_closest_t = 1
graph_plot_grid = 1
# graph_colormap =
exec('graph_colormap=cm.'+str(graph_settings['colormap']))

######################################################################
# add some global vars to dictionaries
graph_settings['stride'] = graph_stride
graph_limits['xmin'] = graph_xmin
graph_limits['xmax'] = graph_xmax
graph_limits['ymin'] = graph_ymin
graph_limits['ymax'] = graph_ymax
graph_limits['vmin'] = graph_vmin
graph_limits['vmax'] = graph_vmax

######################################################################
# functions

# setup ax for either 2d or 3d plots
def setup_axes(fig, graph_3dOn, ax=None):
  if ax != None:
    fig.delaxes(ax)
  if graph_3dOn == 0:
    # make 2d ax to plot graph
    ax = fig.add_subplot(111)
    ax.set_xlim(graph_xmin, graph_xmax)
    ax.set_ylim(graph_vmin, graph_vmax)
  else:
    # make 3d ax to plot graph
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(graph_xmin, graph_xmax)
    ax.set_ylim(graph_ymin, graph_ymax)
    ax.set_zlim(graph_vmin, graph_vmax)
  return ax

# plot into ax at time t
def axplot2d_at_time(filelist, canvas, ax, t):
  global graph_clear_on_replot
  xlim = ax.get_xlim()
  ylim = ax.get_ylim()
  xscale = ax.get_xscale()
  yscale = ax.get_yscale()
  if graph_clear_on_replot == 1:
    ax.clear()
  ct = graph_plot_closest_t
  for i in range(0, len(filelist.file)):
    f = filelist.file[i]
    msiz=graph_linemarkersizes['#'+str(i)]
    if graph_plot_scatter == 1:
      mark=str(graph_linemarkers['#'+str(i)])
      if mark == '' or mark == 'None':
        mark='o'
      ax.scatter(f.data.getx(t,ct), f.data.getv(t,ct), label=f.name,
                 color=graph_linecolors['#'+str(i)], marker=mark)
    elif str(graph_linewidths['#'+str(i)]) == '':
      ax.plot(f.data.getx(t,ct), f.data.getv(t,ct), label=f.name,
              color=graph_linecolors['#'+str(i)],
              linestyle=graph_linestyles['#'+str(i)],
              marker=graph_linemarkers['#'+str(i)],
              markersize=msiz)
    else:
      ax.plot(f.data.getx(t,ct), f.data.getv(t,ct), label=f.name,
              color=graph_linecolors['#'+str(i)],
              linewidth=float(graph_linewidths['#'+str(i)]),
              linestyle=graph_linestyles['#'+str(i)],
              marker=graph_linemarkers['#'+str(i)],
              markersize=msiz)
  ax.set_xlim(xlim)
  ax.set_ylim(ylim)
  ax.set_xscale(xscale)
  ax.set_yscale(yscale)
  if graph_labelsOn == 1:
    ax.set_xlabel(graph_labels['x-axis'], fontsize=graph_labels['fontsize'])
    ax.set_ylabel(graph_labels['v-axis'], fontsize=graph_labels['fontsize'])
    ax.set_title(graph_labels['title'])
    tf = graph_labels['timeformat']
    if len(tf) > 0:
      tstr = tf % t
      ax.set_title(tstr, loc='right')
  if graph_legendOn == 1:
    ax.legend(fontsize=graph_legend['fontsize'], loc=graph_legend['loc'],
              ncol=graph_legend['ncol'])
  if graph_plot_grid == 1:
    ax.grid(True)

# plot into ax at time t, in 3d
def axplot3d_at_time(filelist, canvas, ax, t):
  global graph_stride
  global graph_colormap
  global graph_clear_on_replot
  xlim = ax.get_xlim()
  ylim = ax.get_ylim()
  zlim = ax.get_zlim()
  xscale = ax.get_xscale()
  yscale = ax.get_yscale()
  zscale = ax.get_zscale()
  if graph_clear_on_replot == 1:
    ax.clear()
  ct = graph_plot_closest_t
  for i in range(0, len(filelist.file)):
    f = filelist.file[i]
    blocks = f.data.getblocks(t)
    if blocks < 2 and graph_plot_surface == 1:
      print('3D plot will work only with wireframe, because input data had no empty lines.')
    reshaper = (blocks, -1)
    x=np.reshape(f.data.getx(t,ct), reshaper)
    y=np.reshape(f.data.gety(t,ct), reshaper)
    v=np.reshape(f.data.getv(t,ct), reshaper)
    #print(x,y,v)
    if graph_plot_surface == 1:
      if str(graph_settings['colormap']) == '':
        ax.plot_surface(x,y, v, rstride=graph_stride,cstride=graph_stride,
                        label=f.name, color=graph_linecolors['#'+str(i)],
                        antialiased=int(graph_settings['antialiased']),
                        shade=int(graph_settings['shade']),
                        linewidth=float(graph_settings['linewidth']),
                        edgecolor=graph_settings['edgecolor'])
      else:
        ax.plot_surface(x,y, v, rstride=graph_stride,cstride=graph_stride,
                        label=f.name, cmap=graph_colormap,
                        antialiased=int(graph_settings['antialiased']),
                        shade=int(graph_settings['shade']),
                        linewidth=float(graph_settings['linewidth']),
                        edgecolor=graph_settings['edgecolor'])
    else:
      if graph_plot_scatter == 1:
        mark=str(graph_linemarkers['#'+str(i)])
        if mark == '' or mark == 'None':
          mark='o'
        ax.scatter(x,y, v, label=f.name,
                   color=graph_linecolors['#'+str(i)], marker=mark)
      else:
        ax.plot_wireframe(x,y, v, rstride=graph_stride,cstride=graph_stride,
                          label=f.name, color=graph_linecolors['#'+str(i)])
  # this does not seem to work in 3d:
  #ax.set_xlim(xlim)
  #ax.set_ylim(ylim)
  #ax.set_zlim(zlim)
  #ax.set_xscale(xscale)
  #ax.set_yscale(yscale)
  #ax.set_zscale(zscale)
  ax.set_xlim(graph_xmin, graph_xmax)
  ax.set_ylim(graph_ymin, graph_ymax)
  ax.set_zlim(graph_vmin, graph_vmax)
  if graph_labelsOn == 1:
    ax.set_xlabel(graph_labels['x-axis'], fontsize=graph_labels['fontsize'])
    ax.set_ylabel(graph_labels['y-axis'], fontsize=graph_labels['fontsize'])
    ax.set_zlabel(graph_labels['v-axis'], fontsize=graph_labels['fontsize'])
    ax.set_title(graph_labels['title'])
    tf = graph_labels['timeformat']
    if len(tf) > 0:
      tstr = tf % t
      ax.set_title(tstr, loc='right')
  # Note: legend does not work for surface. Is matplotlib broken???
  if graph_legendOn == 1 and graph_plot_surface == 0:
    ax.legend(fontsize=graph_legend['fontsize'], loc=graph_legend['loc'],
              ncol=graph_legend['ncol'])

def replot():
  global filelist
  global canvas
  global ax
  global graph_time
  global graph_3dOn
  global graph_axis_on
  if graph_3dOn == 0:
    axplot2d_at_time(filelist, canvas, ax, graph_time)
  else:
    axplot3d_at_time(filelist, canvas, ax, graph_time)
  # print(ax.xaxis.tick_top())
  if graph_axis_on == 0:
    ax.set_axis_off()
  canvas.draw()

# callbacks for some events
def update_graph_time_entry():
  global tentry
  global graph_time
  tentry.delete(0, END)
  tentry.insert(0, str(graph_time))
  replot()

def draw_legend():
  ax.legend(fontsize=10)  # ax.legend() # (fontsize=8)
  canvas.draw()

def toggle_log_xscale():
  global ax
  if ax.get_xscale() == 'linear':
    ax.set_xscale('log')
  else:
    ax.set_xscale('linear')
  # since matplotlib 3.6.3 (and possibly earlier)
  # canvas.draw()
  # doesn't plot correctly anymore, so now we call replot()
  replot()

def toggle_log_yscale():
  global ax
  if ax.get_yscale() == 'linear':
    ax.set_yscale('log')
  else:
    ax.set_yscale('linear')
  # since matplotlib 3.6.3 (and possibly earlier)
  # canvas.draw()
  # doesn't plot correctly anymore, so now we call replot()
  replot()

def toggle_plot_grid():
  global graph_plot_grid
  if graph_plot_grid == 1:
    graph_plot_grid = 0
  else:
    graph_plot_grid = 1
  replot()

def toggle_axis_on():
  global graph_axis_on
  if graph_axis_on == 1:
    graph_axis_on = 0
  else:
    graph_axis_on = 1
  replot()

def toggle_2d_3d():
  global fig
  global graph_3dOn
  global ax
  if graph_3dOn == 1:
    graph_3dOn = 0
  else:
    graph_3dOn = 1
  ax = setup_axes(fig, graph_3dOn, ax)
  replot()

def toggle_wireframe_surface():
  global fig
  global graph_3dOn
  global ax
  global graph_plot_surface
  global graph_plot_scatter
  if graph_plot_surface == 1:
    graph_plot_surface = 0
  else:
    graph_plot_surface = 1
  # ax = setup_axes(fig, graph_3dOn, ax)
  replot()

def toggle_wireframe_scatter():
  global fig
  global graph_3dOn
  global ax
  global graph_plot_surface
  global graph_plot_scatter
  if graph_plot_surface == 1:
    graph_plot_scatter = 0
    graph_plot_surface = 0
  if graph_plot_scatter == 1:
    graph_plot_scatter = 0
  else:
    graph_plot_scatter = 1
  # ax = setup_axes(fig, graph_3dOn, ax)
  replot()

def toggle_labels():
  global graph_labelsOn
  if graph_labelsOn == 1:
    graph_labelsOn = 0
  else:
    graph_labelsOn = 1
  replot()

def toggle_legend():
  global graph_legendOn
  if graph_legendOn == 1:
    graph_legendOn = 0
  else:
    graph_legendOn = 1
  replot()

def toggle_clear_on_replot():
  global graph_clear_on_replot
  if graph_clear_on_replot == 1:
    graph_clear_on_replot = 0
  else:
    graph_clear_on_replot = 1
  replot()

def toggle_plot_closest_t():
  global graph_plot_closest_t
  if graph_plot_closest_t == 1:
    graph_plot_closest_t = 0
  else:
    graph_plot_closest_t = 1
  replot()

def BT1_callback(event):
  print("clicked at", event.x, event.y)

def not_implemented():
  print("not implemented yet!")

def set_graph_time(event, ent):
  global graph_time
  global graph_timeindex
  global graph_timelist
  t = ent.get()
  graph_timeindex = tdata.geti_from_t(graph_timelist, float(t))
  graph_time = graph_timelist[graph_timeindex]
  replot()

def set_graph_delay(event, ent):
  global graph_delay 
  graph_delay = float(ent.get())

def min_graph_time():
  global graph_time
  global graph_timeindex
  global graph_timelist
  graph_timeindex = 0
  graph_time = graph_timelist[graph_timeindex]
  update_graph_time_entry()

def max_graph_time():
  global graph_time
  global graph_timeindex
  global graph_timelist
  graph_timeindex = len(graph_timelist)-1
  graph_time = graph_timelist[graph_timeindex]
  update_graph_time_entry()

def inc_graph_time():
  global graph_time
  global graph_timeindex
  global graph_timelist
  if graph_timeindex<len(graph_timelist)-1:
    graph_timeindex += 1
  graph_time = graph_timelist[graph_timeindex]
  update_graph_time_entry()

def dec_graph_time():
  global graph_time
  global graph_timeindex
  global graph_timelist
  if graph_timeindex>0:
    graph_timeindex -= 1
  graph_time = graph_timelist[graph_timeindex]
  update_graph_time_entry()

def play_graph_time():
  global graph_time
  global graph_timeindex
  global graph_timelist
  inc_graph_time()
  if graph_timeindex<len(graph_timelist)-1:
    play_id = root.after(int(graph_delay), play_graph_time)
  else:
    play_id = root.after(int(graph_delay), update_graph_time_entry)
  def cancel_callback(event):
    root.after_cancel(play_id)
  root.bind("<Button-1>", cancel_callback)

def start_play_graph_time():
  global graph_time
  global graph_timeindex
  global graph_timelist
  i1 = graph_timeindex
  if i1>=len(graph_timelist)-1:
    i1 = 0
  graph_timeindex = i1
  graph_time = graph_timelist[graph_timeindex]
  update_graph_time_entry()
  root.after(int(graph_delay), play_graph_time)

def open_and_plot_file():
  open_file()
  replot()

def save_movieframes():
  global graph_time
  global graph_timeindex
  global graph_timelist
  global fig
  # get filename
  fname = filedialog.asksaveasfilename(initialfile='frame.png',
                          title='Enter base movie frame name with extension')
  if len(fname) == 0:  # if user presses cancel fname is () or '', so exit
    return
  p = fname.rfind('.')
  if p >= 0:
    ext  = fname[p:]
    base = fname[:p]
  else:
    ext  = ''
    base = fname
  # first and last time index
  i1 = 0
  i2 = len(graph_timelist)
  # format (something like '%.6d') we use to print time index into filename
  fmt = '%.'
  fmt += '%d' % int( np.log10(i2)+1 )
  fmt += 'd'
  # loop over time indices
  for graph_timeindex in range(i1, i2):
    graph_time = graph_timelist[graph_timeindex]
    update_graph_time_entry()
    tstr = fmt % graph_timeindex
    name = base + '_' + tstr + ext
    canvas.print_figure(name)
    if graph_timeindex == i1:
      name1 = name
  movie_message(name1, name)

def movie_message(name1, name2):
  top1 = Tk()
  top1.wm_title("Movie Frame Info")
  str =  ' Movie Frames have been saved in the files: \n'
  str += '   ' + name1 + ' ... ' + name2 + ' \n'
  l1 = Label(master=top1, text=str)
  l1.pack(side=TOP, expand=1)
  button = Button(top1, text="Close", command=top1.destroy)
  button.pack(side=TOP)
  top1.mainloop()

def about():
  top1 = Tk()
  top1.wm_title("About tgraph")
  str =  " tgraph " + tgraph_version + " \n\n"
  str += "   Produce quick 2D or 3D graphs from files given on the command line. \n"
  str += "   Read the file tgraph.txt for help. \n\n"
  str += "   Copyright (C) 2015 Wolfgang Tichy. \n"
  l1 = Label(master=top1, text=str)
  l1.pack(side=TOP, expand=1)
  button = Button(top1, text="Close", command=top1.destroy)
  button.pack(side=TOP)
  top1.mainloop()

def help():
  global tgraph_txt_file
  top1 = Tk()
  top1.wm_title("tgraph help")
  str =  "   tgraph.txt for tgraph " + tgraph_version + ","
  str += "   Copyright (C) 2015 Wolfgang Tichy.  "
  l1 = Label(master=top1, text=str)
  l1.pack(side=TOP, expand=1)
  scrollbar = Scrollbar(master=top1)
  scrollbar.pack(side=RIGHT, fill=Y)
  text = Text(master=top1, wrap=WORD, yscrollcommand=scrollbar.set)
  tgraph_txt = "file tgraph.txt not found!"
  with open(tgraph_txt_file, 'r') as f:
    tgraph_txt = f.read()
  text.insert("1.0", tgraph_txt)
  # text.config(state=DISABLED) # no editing in text window
  text.pack()
  scrollbar.config(command=text.yview)
  button = Button(top1, text="Close", command=top1.destroy)
  button.pack(side=TOP)
  top1.mainloop()

# simple dialog that opens window where we can enter values for a dictionary
class WTdialog:
    # init input form, form is a dict. containing labels and values
  def __init__(self, title, formdict):
    self.input = {}     # init input dict.
    self.Entry = {}     # init tk Entry dict.
    self.top = Toplevel(root)  # root is parent and can now wait for self.top
    self.top.wm_title(title)
    f0 = Frame(master=self.top)
    f0.pack(side=TOP, expand=1)
    row = 0
    for key in sorted(formdict):
      label = key
      entry = formdict[key]
      l1 = Label(master=f0, text=label)
      l1.grid(row=row, column=0)
      e1 = Entry(master=f0, width=80)
      e1.grid(row=row, column=1)
      e1.delete(0, END)
      e1.insert(0, entry)
      # make duplicate of formdict and save tk Entry objects
      self.input[label] = entry
      self.Entry[label] = e1
      row += 1
    # add "Apply" button
    button = Button(self.top, text="Apply", command=self.apply_changes)
    button.pack(side=TOP)
    # wait for window self.top
    root.wait_window(self.top)

  def get_input_values(self):
    for key in self.input:
      self.input[key] = self.Entry[key].get()

  def apply_changes(self):
    self.get_input_values()
    #self.top.quit()
    self.top.destroy()

# use WTdialog to reset some limits
def input_graph_limits():
  global fig
  global graph_3dOn
  global ax
  global graph_limits  # dict. with options
  global graph_xmin
  global graph_xmax
  global graph_ymin
  global graph_ymax
  global graph_vmin
  global graph_vmax
  # get graph_limits
  dialog = WTdialog("tgraph Limits", graph_limits)
  # now get the user input back
  graph_limits = dialog.input
  graph_xmin = float(graph_limits['xmin'])
  graph_xmax = float(graph_limits['xmax'])
  graph_ymin = float(graph_limits['ymin'])
  graph_ymax = float(graph_limits['ymax'])
  graph_vmin = float(graph_limits['vmin'])
  graph_vmax = float(graph_limits['vmax'])
  # change axes and then plot again
  ax = setup_axes(fig, graph_3dOn, ax)
  replot()

# set graph_limits dict. from graph_xmin, graph_xmax, ...
def set_graph_limits():
  global graph_limits  # dict. with options
  global graph_xmin
  global graph_xmax
  global graph_ymin
  global graph_ymax
  global graph_vmin
  global graph_vmax
  graph_limits['xmin'] = graph_xmin
  graph_limits['xmax'] = graph_xmax
  graph_limits['ymin'] = graph_ymin
  graph_limits['ymax'] = graph_ymax
  graph_limits['vmin'] = graph_vmin
  graph_limits['vmax'] = graph_vmax

# use WTdialog to set xcols
def input_graph_xcolumns():
  global filelist
  global graph_xmin
  global graph_xmax
  global fig
  global graph_3dOn
  global ax
  xcoldict = {}
  for i in range(0, len(filelist.file)):
    xcoldict['#'+str(i)] = filelist.file[i].data.get_xcol0()+1
  dialog = WTdialog("tgraph x-Column", xcoldict)
  xcoldict = dialog.input
  for i in range(0, len(filelist.file)):
    filelist.file[i].data.set_xcols(int(xcoldict['#'+str(i)])-1)
  graph_xmin = tdata.inf_to_1e300(filelist.minx())
  graph_xmax = tdata.inf_to_1e300(filelist.maxx())
  set_graph_limits()
  print('(xmin, xmax) =', '(', graph_xmin, ',', graph_xmax, ')')
  ax = setup_axes(fig, graph_3dOn, ax)
  replot()

# use WTdialog to set ycols
def input_graph_ycolumns():
  global filelist
  global graph_ymin
  global graph_ymax
  global fig
  global graph_3dOn
  global ax
  ycoldict = {}
  for i in range(0, len(filelist.file)):
    ycoldict['#'+str(i)] = filelist.file[i].data.get_ycol0()+1
  dialog = WTdialog("tgraph y-Column", ycoldict)
  ycoldict = dialog.input
  for i in range(0, len(filelist.file)):
    filelist.file[i].data.set_ycols(int(ycoldict['#'+str(i)])-1)
  graph_ymin = tdata.inf_to_1e300(filelist.miny())
  graph_ymax = tdata.inf_to_1e300(filelist.maxy())
  set_graph_limits()
  print('(ymin, ymax) =', '(', graph_ymin, ',', graph_ymax, ')')
  ax = setup_axes(fig, graph_3dOn, ax)
  replot()

# use WTdialog to set vcols
def input_graph_vcolumns():
  global filelist
  global graph_vmin
  global graph_vmax
  global fig
  global graph_3dOn
  global ax
  vcoldict = {}
  for i in range(0, len(filelist.file)):
    vcoldict['#'+str(i)] = filelist.file[i].data.get_vcol0()+1
  dialog = WTdialog("tgraph v-Column", vcoldict)
  vcoldict = dialog.input
  for i in range(0, len(filelist.file)):
    filelist.file[i].data.set_vcols(int(vcoldict['#'+str(i)])-1)
  graph_vmin = tdata.inf_to_1e300(filelist.minv())
  graph_vmax = tdata.inf_to_1e300(filelist.maxv())
  set_graph_limits()
  print('(vmin, vmax) =', '(', graph_vmin, ',', graph_vmax, ')')
  ax = setup_axes(fig, graph_3dOn, ax)
  replot()

# use WTdialog to reset some settings
def input_graph_settings():
  global fig
  global graph_3dOn
  global ax
  global graph_settings  # dict. with options
  global graph_colormap
  global graph_stride
  # get graph_settings
  dialog = WTdialog("tgraph Settings", graph_settings)
  # now get the user input back
  graph_settings = dialog.input
  mpl.rcParams['lines.linewidth'] = graph_settings['linewidth']
  if str(graph_settings['colormap']) != '':
    exec('global graph_colormap;' +
         'graph_colormap = cm.' + str(graph_settings['colormap']))
  graph_stride = int(graph_settings['stride'])
  # change axes and then plot again
  replot()

# use WTdialog to reset some labels
def input_graph_labels():
  global graph_labelsOn
  global graph_labels  # dict. with options
  # get graph_labels
  dialog = WTdialog("tgraph Labels", graph_labels)
  # now get the user input back
  graph_labels = dialog.input
  mpl.rcParams['font.size'] = graph_labels['fontsize']
  # since we edited the labels switch them on now
  graph_labelsOn = 1
  replot()

# use WTdialog to reset legend
def input_graph_legend():
  global filelist
  global graph_legendOn
  global graph_legend # dict. with options
  # for legend
  # convert graph_legend['loc'] to str, which is undone below
  graph_legend['loc'] = str(graph_legend['loc'])
  # now update graph_legend
  dialog = WTdialog("tgraph Legend", graph_legend)
  graph_legend = dialog.input
  # save names
  for i in range(0, len(filelist.file)):
    f = filelist.file[i]
    f.name = graph_legend['#'+str(i)]
  # Check what loc has. It could be a string, an int, or a coordinate tuple.
  s = graph_legend['loc']
  s = s.rstrip()
  s = s.lstrip()
  graph_legend['loc'] = s
  if s.isdigit():
    graph_legend['loc'] = int(s)
  else:
    pos = s.find(',')
    if pos >= 0:
      s = s.replace('[', ' ')
      s = s.replace(']', ' ')
      s = s.replace('(', ' ')
      s = s.replace(')', ' ')
      l = s.split(',')
      graph_legend['loc'] = [ float(l[0]), float(l[1]) ]
  # make graph_legend['ncol'] an int
  graph_legend['ncol'] = int(graph_legend['ncol'])
  # set some things in mpl.rcParams
  mpl.rcParams['legend.fancybox']     = graph_legend['fancybox']
  mpl.rcParams['legend.shadow']       = graph_legend['shadow']
  mpl.rcParams['legend.frameon']      = graph_legend['frameon']
  mpl.rcParams['legend.framealpha']   = graph_legend['framealpha']
  mpl.rcParams['legend.handlelength'] = graph_legend['handlelength']
  # since we edited the legend switch it on now
  graph_legendOn = 1
  replot()

# use WTdialog to reset legend
def edit_mpl_rcParams():
  dialog = WTdialog("mpl.rcParams", mpl.rcParams)
  mpl.rcParams = dialog.input
  replot()

# use WTdialog to reset line colors
def input_graph_linecolors():
  global filelist
  global graph_linecolors # dict. with options
  dialog = WTdialog("tgraph Line Colors", graph_linecolors)
  graph_linecolors = dialog.input
  replot()

# use WTdialog to reset line colors
def input_graph_linestyles():
  global filelist
  global graph_linestyles # dict. with options
  dialog = WTdialog("tgraph Line Styles", graph_linestyles)
  graph_linestyles = dialog.input
  replot()

# use WTdialog to reset line markers
def input_graph_linemarkers():
  global filelist
  global graph_linemarkers # dict. with options
  dialog = WTdialog("tgraph Line Markers", graph_linemarkers)
  graph_linemarkers = dialog.input
  replot()

# use WTdialog to reset line markers
def input_graph_linemarkersizes():
  global filelist
  global graph_linemarkersizes # dict. with options
  dialog = WTdialog("tgraph Line Markersizes", graph_linemarkersizes)
  graph_linemarkersizes = dialog.input
  replot()

# use WTdialog to reset line widths
def input_graph_linewidths():
  global filelist
  global graph_linewidths # dict. with options
  dialog = WTdialog("tgraph Line Widths", graph_linewidths)
  graph_linewidths = dialog.input
  replot()

# use WTdialog to do transformations on columns
def input_graph_coltrafos():
  global filelist
  global graph_coltrafos # dict. with trafos
  dialog = WTdialog(
    "tgraph Column Transformations, "
    "e.g. c[3] = 2*c[2] + sin(t) +  D(c[2])/D(c[1])",
    graph_coltrafos)
  graph_coltrafos = dialog.input
  # print(graph_coltrafos)
  for i in range(0, len(filelist.file)):
    f = filelist.file[i]
    trafo = str(graph_coltrafos['#'+str(i)])
    if trafo == '':
      continue
    else:
      print("transform", '#'+str(i)+':', trafo)
      f.data.transform_col(trafo, c_index_shift=1)
  replot()

######################################################################
# except for root window all tk stuff follows below
######################################################################
# make menu bar
menubar = Menu(root)

# create a pulldown menu, and add it to the menu bar
filemenu = Menu(menubar, tearoff=0)
#filemenu.add_command(label="Open", command=not_implemented)
filemenu.add_command(label="Open File", command=open_and_plot_file)
filemenu.add_command(label="Save Movie Frames", command=save_movieframes)
#filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.destroy)
menubar.add_cascade(label="File", menu=filemenu)

# create more pulldown menus
optionsmenu = Menu(menubar, tearoff=0)
optionsmenu.add_command(label="Toggle Timeframe update/add", command=toggle_clear_on_replot)
optionsmenu.add_command(label="Toggle Timeframe closest/exact", command=toggle_plot_closest_t)
optionsmenu.add_command(label="Toggle Axis on/off", command=toggle_axis_on)
optionsmenu.add_command(label="Toggle log/lin x", command=toggle_log_xscale)
optionsmenu.add_command(label="Toggle log/lin y", command=toggle_log_yscale)
optionsmenu.add_command(label="Toggle Grid on/off", command=toggle_plot_grid)
optionsmenu.add_command(label="Toggle Line/Scatter",
                        command=toggle_wireframe_scatter)
optionsmenu.add_command(label="Toggle 2D/3D", command=toggle_2d_3d)
optionsmenu.add_command(label="Toggle 3D-Surface",
                        command=toggle_wireframe_surface)
optionsmenu.add_command(label="Toggle Labels", command=toggle_labels)
optionsmenu.add_command(label="Toggle Legend", command=toggle_legend)
#optionsmenu.add_command(label="Show Legend", command=draw_legend)
menubar.add_cascade(label="Options", menu=optionsmenu)

settingsmenu = Menu(menubar, tearoff=0)
settingsmenu.add_command(label="Select x-Columns", command=input_graph_xcolumns)
settingsmenu.add_command(label="Select y-Columns", command=input_graph_ycolumns)
settingsmenu.add_command(label="Select v-Columns", command=input_graph_vcolumns)
settingsmenu.add_command(label="Edit Limits", command=input_graph_limits)
settingsmenu.add_command(label="Edit Labels", command=input_graph_labels)
settingsmenu.add_command(label="Edit Legend", command=input_graph_legend)
settingsmenu.add_command(label="Graph Settings", command=input_graph_settings)
#settingsmenu.add_command(label="Edit rcParams", command=edit_mpl_rcParams)
menubar.add_cascade(label="Settings", menu=settingsmenu)

linesmenu = Menu(menubar, tearoff=0)
linesmenu.add_command(label="Edit Line Colors",  command=input_graph_linecolors)
linesmenu.add_command(label="Edit Line Styles",  command=input_graph_linestyles)
linesmenu.add_command(label="Edit Line Markers", command=input_graph_linemarkers)
linesmenu.add_command(label="Edit Line Markersizes", command=input_graph_linemarkersizes)
linesmenu.add_command(label="Edit Line Widths",  command=input_graph_linewidths)
menubar.add_cascade(label="Lines", menu=linesmenu)


transformationsmenu = Menu(menubar, tearoff=0)
transformationsmenu.add_command(label="Transform Columns",
                                command=input_graph_coltrafos)
menubar.add_cascade(label="Transformations", menu=transformationsmenu)


# create help pulldown menu
helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="About", command=about)
helpmenu.add_command(label="Read help in tgraph.txt", command=help)
menubar.add_cascade(label="Help", menu=helpmenu)

# display the menu
root.config(menu=menubar)


# make a frame where we put time step controls
topframe = Frame(root)
topframe.pack(side=TOP, expand=0)

# entries for time 
tl = Label(master=topframe, text="Time")
tl.pack(side=LEFT, expand=0)
tentry = Entry(master=topframe, width=22)
tentry.bind('<Return>', lambda event, ent=tentry: set_graph_time(event, ent))
tentry.bind('<Deactivate>', lambda event, ent=tentry: set_graph_time(event, ent))
tentry.pack(side=LEFT, expand=1)
tentry.delete(0, END)
tentry.insert(0, graph_time)

# add buttons for player
sb = Button(master=topframe, text='<<', width=3, command=min_graph_time)
sb.pack(side=LEFT, expand=1)
bb = Button(master=topframe, text='<', width=3, command=dec_graph_time)
bb.pack(side=LEFT, expand=1)
pb = Button(master=topframe, text='Play', width=5, command=start_play_graph_time)
pb.pack(side=LEFT, expand=1)
fb = Button(master=topframe, text='>', width=3, command=inc_graph_time)
fb.pack(side=LEFT, expand=1)
eb = Button(master=topframe, text='>>', width=3, command=max_graph_time)
eb.pack(side=LEFT, expand=1)

# entries for delay
dl = Label(master=topframe, text="  Delay")
dl.pack(side=LEFT, expand=1)
de = Entry(master=topframe, width=4)
de.bind('<Return>', lambda event, ent=de: set_graph_delay(event, ent))
de.bind('<Leave>', lambda event, ent=de: set_graph_delay(event, ent))
de.pack(side=LEFT, expand=1)
de.delete(0, END)
de.insert(0, "1")

######################################################################
# make figure fig
fig = Figure(figsize=(7.25, 7), dpi=85)

# Use matplotlib to make a tk.DrawingArea of fig and show it.
# This need needs to come before making ax by: ax = Axes3D(fig)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

# setup the axes
ax = setup_axes(fig, graph_3dOn, None)

# make matplotlib toolbar
toolbar = NavToolbar(canvas, root)
toolbar.update()
#canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=1)
canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

replot()

######################################################################
#print('times =', graph_timelist)
print('(tmin, tmax) =', '(', filelist.mintime(), ',', filelist.maxtime(), ')')
print('(xmin, xmax) =', '(', graph_xmin, ',', graph_xmax, ')')
print('(ymin, ymax) =', '(', graph_ymin, ',', graph_ymax, ')')
print('(vmin, vmax) =', '(', graph_vmin, ',', graph_vmax, ')')


######################################################################
# go into tkinter's main loop and wait for events
root.mainloop()
