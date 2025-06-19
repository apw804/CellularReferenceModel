"""
cellular_reference_model_10d.py
-------------------------------
A cellular reference model for simulating wireless cellular networks, including cell and user equipment (UE) layout, path loss, and path gain calculations. 
Supports multiple path loss models (free-space, RMa, UMa, LOS/NLOS), flexible cell and UE layouts, and visualization utilities.

Classes
-------
Energy_reference_model_layout
  Generates and manages the spatial layout of cells and UEs for the simulation, supporting hexagonal and Poisson Point Process (PPP) layouts.
CRMParameters
  Dataclass for holding configuration parameters for the Cellular Reference Model (CRM), including cell/UE counts, radii, layout types, and pathloss model selection.
CellularReferenceModel
  Main class for simulating the cellular network, computing distance and path gain matrices, and providing plotting and UE re-generation utilities.
Plot_CRM_layout
  Visualisation class for plotting the cellular layout, UEs, Voronoi diagrams, and additional overlays such as pathloss circles.

Notes
-----
- Path loss models supported: 'free-space', 'RMa-LOS', 'RMa-NLOS', 'UMa-LOS', 'UMa-NLOS'.
- Visualisation supports custom cell/UE images, Voronoi overlays, and pathloss contour circles.
- Designed for research and educational use in cellular network modeling.
"""

from sys import exit
from os import getuid
from pwd import getpwuid
from itertools import product
from dataclasses import dataclass
from math import log1p, sqrt, pi, cos, sin
from platform import system as platform_system

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import ticker

from RMa_pathloss_model_03 import RMa_pathloss
from UMa_pathloss_model_02 import UMa_pathloss
from fig_timestamp_01 import fig_timestamp
from hexagon_lattice_generator_02 import hexagon_lattice_generator
from color_text_00 import Color_text

__version__='10d'
color_text=Color_text()

class Energy_reference_model_layout:
  """
  Layout generator for cellular reference models with configurable cell and UE (User Equipment) distributions.

  Parameters
  ----------
  n_cells : int
    Number of cells in the layout.
  n_ues : int
    Number of UEs (mean number if UE_layout='PPP').
  cell_layout : str, optional
    Cell layout type ('hex' for hexagonal lattice). Default is 'hex'.
  UE_layout : str, optional
    UE placement strategy ('PPP' for Poisson Point Process, 'uniform' for uniform distribution). Default is 'PPP'.
  UE_radius : float, optional
    Radius for UE placement. If 0.0, uses cell_rmax. Default is 0.0.
  radius_boost : float, optional
    Additional radius for PPP UE placement. Default is 2.5e3.
  cell_radius_m : float, optional
    Radius of each cell in meters. Default is 5e3.
  seed : int, optional
    Random seed for reproducibility. Default is 0.
  verbose : bool, optional
    If True, prints detailed debug information. Default is False.

  Attributes
  ----------
  cells : ndarray
    Array of cell center coordinates, shape (n_cells, 2).
  ues : ndarray
    Array of UE coordinates, shape (n_ues_actual, 2).
  cell_rmax : float
    Maximum distance from origin to any cell center.
  system_rmax : float
    Maximum system radius (cell_rmax + radius_boost).
  cell_area : float
    Area covered by the cells.
  system_area : float
    Total area considered for UE placement.
  n_ues_actual : int
    Actual number of UEs generated (for PPP layout).

  Methods
  -------
  get_cells()
    Returns the array of cell coordinates.
  get_ues()
    Returns the array of UE coordinates.
  get_nues()
    Returns the number of UEs.
  generate_new_UE_layout(seed)
    Regenerates the UE layout with a new random seed.
  """

  def __init__(self,n_cells,n_ues,cell_layout='hex',UE_layout='PPP', UE_radius=0.0, radius_boost=2.5e3,cell_radius_m=5e3,seed=0,verbose=False):
    """
    Parameters
    ----------
    n_cells : int
      Number of cells in the layout.
    n_ues : int or float
      Number of user equipments (UEs). For 'PPP' layout, this is the mean number.
    cell_layout : str, optional
      Type of cell layout. Default is 'hex'.
    UE_layout : str, optional
      Type of UE layout. Default is 'PPP'.
    UE_radius : float, optional
      Radius for UE placement. Default is 0.0.
    radius_boost : float, optional
      Multiplier for the PPP circle radius. Default is 2.5e3.
    cell_radius_m : float, optional
      Cell radius in meters. Default is 5e3.
    seed : int, optional
      Random seed for reproducibility. Default is 0.
    verbose : bool, optional
      If True, enables verbose output. Default is False.
    """
    self.n_cells = n_cells
    self.n_ues = n_ues
    self.cell_layout = cell_layout
    self.UE_layout = UE_layout
    self.UE_radius = UE_radius
    self.radius_boost = radius_boost
    self.verbose = verbose
    self.cell_radius_m = cell_radius_m
    self.init_cell_layout(seed)
    self.init_UE_layout(seed)

  def init_cell_layout(self,seed=0):
    """
    Initialize the positions and layout of cells in the system.

    Parameters
    ----------
    seed : int, optional
      Random seed for reproducibility (default is 0).

    Notes
    -----
    Currently, only the 'hex' cell layout is implemented. The method sets up cell positions,
    computes the maximum cell radius, system radius, and corresponding areas.
    """
    if self.verbose: print(color_text.Blue(f'Energy_reference_model_layout.init_cell_layout: n_cells={self.n_cells} (layout="{self.cell_layout}") n_ues={self.n_ues} (layout="{self.UE_layout}") seed={seed}'))
    self.cells=np.empty((self.n_cells,2))
    if self.cell_layout=='hex':
      hlg=hexagon_lattice_generator()
      for i in range(self.n_cells): self.cells[i]=next(hlg)
    else: 
      print(color_text.Red(f'cell_layout={self.cell_layout} not implemented, quitting 😡'))
      exit(1)
    self.cells[:,:2]=self.cell_radius_m*self.cells[:,:2]
    self.cell_rmax=np.max(np.linalg.norm(self.cells,axis=1))
    self.cell_rmax=max(self.cell_radius_m,self.cell_rmax) 
    self.system_rmax=self.radius_boost+self.cell_rmax
    self.cell_area  =np.pi*self.cell_rmax**2
    self.system_area=np.pi*self.system_rmax**2
    if self.verbose: print(color_text.Blue(f'Energy_reference_model_layout.init_cell_layout: cell_rmax={self.cell_rmax:g} system_rmax={self.system_rmax:g} self.system_area={self.system_area:g}'))

  def init_UE_layout(self,seed=0):
    """
    Initialize the layout of User Equipments (UEs) in the simulation area.

    Parameters
    ----------
    seed : int, optional
      Seed for the random number generator (default is 0).

    Notes
    -----
    The UE positions are generated according to the specified `UE_layout` attribute.
    Supported layouts:
      - 'uniform': UEs are distributed uniformly within a circle of radius `UE_radius` (or `cell_rmax` if `UE_radius` is 0).
      - 'PPP': UEs are distributed according to a Poisson Point Process (PPP) with intensity λ = n_ues / system_area.
    """
    self.rng=np.random.default_rng(seed)
    if self.UE_radius==0.0:
      R=self.cell_rmax
    else:
      R=self.UE_radius
    if self.UE_layout=='uniform':
      if self.cells.shape==(1,2):
        # We need to place the UEs in the bounds of the simulation area, NOT just the cell locations.
        length= R*np.sqrt(self.rng.uniform(low=0, high=np.power(self.UE_radius, 2), size=self.n_ues))
        angle=self.rng.uniform(0,1,size=self.n_ues)*2*np.pi
        xs=self.cells[0][0]+length*np.cos(angle)
        ys=self.cells[0][1]+length*np.sin(angle)
        self.ues=np.stack((xs,ys), axis=1)
        return
      lengths=R*np.sqrt(self.rng.uniform(low=0, high=1, size=(self.n_ues)))
      thetas=2*np.pi*self.rng.uniform(0,1,size=self.n_ues)
      ue_xs=0.0+lengths*np.cos(thetas) # 0.0=x-origin
      ue_ys=0.0+lengths*np.sin(thetas) # 0.0=y-origin
      self.ues=np.column_stack((ue_xs,ue_ys))
    elif self.UE_layout=='PPP':
      λ=self.n_ues/self.system_area
      self.ues=np.empty((0,2))
      for r,θ in self.poisson_point_process_generator(λ):
        if r>self.system_rmax: break
        xy=np.array([[r*cos(θ),r*sin(θ)]])
        self.ues=np.append(self.ues,xy,axis=0)
      self.n_ues_actual=self.ues.shape[0]
      if self.verbose: print(color_text.Blue(f'Energy_reference_model_layout.init_UE_layout: {self.n_ues} UEs requested, {self.n_ues_actual} actual PPP UE locations generated up to radius={self.system_rmax}, in area {self.system_area:g}, λ={λ:g}'))
    else:
      print(color_text.Red(f'UE_layout={self.UE_layout} not implemented, quitting 😡'))
      exit(1)

  def poisson_point_process_generator(self,λ):
    """ Radial Poisson generator. λ = average points per unit area"""
    a,s,twopi=-1.0/pi/λ,0.0,2.0*pi
    while True:
      s+=log1p(-self.rng.random())
      yield sqrt(a*s),twopi*self.rng.random()

  def get_cells(self): 
    """
    Returns
    -------
    np.ndarray
        Array of cell center coordinates, shape (n_cells, 2).
    """
    return self.cells
  
  def get_ues(self):   
    """
    Returns
    -------
    np.ndarray
        Array of UE coordinates, shape (n_ues_actual, 2).
    """
    return self.ues
  
  def get_nues(self):  
    """
    Returns
    -------
    int
        Number of UEs in the current layout.
    """
    return self.ues.shape[0]
  
  def generate_new_UE_layout(self, seed):
    """
    Generate a new User Equipment (UE) layout using the specified random seed.

    Parameters
    ----------
    seed : int
      Seed for the random number generator to ensure reproducibility.
    """
    self.init_UE_layout(seed)
# END class Energy_reference_model_layout

@dataclass(frozen=False)
class CRMParameters:
  """
  CRMParameters dataclass holds configuration parameters for the Cellular Reference Model (CRM).

  Parameters
  ----------
  n_cells : int
    Number of cells in the simulation.
  n_ues : int
    Number of user equipments (UEs) in the simulation.
  cell_layout : str, optional
    Layout of the cells. Default is 'hex'.
  cell_radius_m : float, optional
    Radius of each cell in meters. Default is 5000.
  UE_layout : str, optional
    Layout of the UEs. Default is 'PPP' (Poisson Point Process).
  UE_radius : float, optional
    Radius for UE placement. Default is 0.0.
  full_cell_PPP : bool, optional
    If True, UEs are distributed over the full cell area using PPP. Default is False.
  pathloss_model_name : str, optional
    Name of the pathloss model to use. Valid values are: 'free-space', 'RMa-LOS', 'RMa-NLOS', 'UMa-LOS', 'UMa-NLOS'. Default is 'free-space'.
  radius_boost : float, optional
    Value added to cell_rmax to compute system_rmax. Default is 2500.
  fc_GHz : float, optional
    Carrier frequency in GHz. Default is 3.5.
  h_UT : float, optional
    Height of the user terminal (UE) in meters. Default is 1.5.
  h_BS : float, optional
    Height of the base station in meters. Default is 20.0.
  LOS : bool, optional
    Line-of-sight condition. Default is True.
  author : str, optional
    Author of the configuration. Default is empty string.
  verbose : bool, optional
    If True, enables verbose output. Default is False.
  """
  n_cells: int
  n_ues: int
  cell_layout: str           = 'hex'
  cell_radius_m: float       = 5e3
  UE_layout: str             = 'PPP'
  UE_radius: float           = 0.0
  full_cell_PPP: bool        = False
  pathloss_model_name: str   = 'free-space'
  radius_boost: float        =  2.5e3 
  fc_GHz: float              =  3.5 
  h_UT: float                =  1.5 
  h_BS: float                = 20.0
  LOS                        = True
  author: str                = ''
  verbose: bool              = False

class CellularReferenceModel:
  """
  Cellular reference model for simulating wireless cellular networks, including cell and user equipment (UE) layout, path loss, and path gain calculations. Supports multiple path loss models (free-space, RMa, UMa, LOS/NLOS).

  Parameters
  ----------
  params : CRMParameters
    Configuration parameters for the cellular model, including number of cells, UEs, layout, radii, and pathloss model.
  sigma_W : float, optional
    Standard deviation of the shadowing (in Watts), by default 2e-13.
  seed : int, optional
    Random seed for reproducibility, by default 0.

  Attributes
  ----------
  params : CRMParameters
    Model parameters.
  sigma_W : float
    Shadowing standard deviation.
  log2 : float
    Precomputed value of log(2).
  erml : Energy_reference_model_layout
    Layout object for cells and UEs.
  seed : int
    Random seed.
  cells : np.ndarray
    Array of cell positions (with heights).
  ues : np.ndarray
    Array of UE positions (with heights).
  n_ues_actual : int
    Actual number of UEs in the layout.
  distance_matrix : np.ndarray
    Matrix of distances between each cell and UE.
  ue_to_cell : np.ndarray
    Index of the nearest cell for each UE.
  pathloss_model : callable
    Function to compute path loss between cell and UE positions.
  pathgain_matrix : np.ndarray
    Matrix of path gains between each cell and UE.

  Methods
  -------
  init_model(seed)
    Initializes the model with the given random seed.
  init_matrices()
    Initializes distance, attachment, and path gain matrices.
  get_pathloss_dB(d)
    Computes the path loss in dB for a given distance.
  generate_new_ue_locations(seed=0)
    Generates a new random UE layout and updates matrices.
  _dbg_get_RMa_pathloss_dB()
    Returns the RMa path loss in dB for debugging.
  plot(...)
    Plots the cellular layout and various overlays.
  get_nues()
    Returns the number of UEs in the current layout.
  """
  def __init__(s, params: CRMParameters, sigma_W: float = 2e-13, seed=0):
    s.params=params
    s.sigma_W=sigma_W
    s.log2=np.log(2.0)
    s.erml=Energy_reference_model_layout(
        n_cells=            s.params.n_cells,
        n_ues=              s.params.n_ues,
        cell_layout=        s.params.cell_layout,
        UE_layout=          s.params.UE_layout,
        UE_radius=          s.params.UE_radius,
        radius_boost=       s.params.radius_boost,
        cell_radius_m=      s.params.cell_radius_m,
        verbose=            s.params.verbose,
      )
    s.seed=seed
    s.init_model(seed)
    
  def init_model(s,seed):
    """
    Initialize the model by retrieving and augmenting cell and UE positions, and initializing matrices.
    """
    s.cells=s.erml.get_cells() # cells indexed by i (in the maths)
    s.ues=s.erml.get_ues()     # UEs   indexed by j
    s.n_ues_actual=s.erml.get_nues()
    s.cells=np.column_stack((s.cells, np.full(s.cells.shape[0], s.params.h_BS)))
    s.ues=np.column_stack((s.ues, np.full(s.n_ues_actual, s.params.h_UT)))
    s.init_matrices()
  
  def init_matrices(s):
    """
    Initialize distance, association, and path gain matrices for the cellular reference model.

    Calculates the distance matrix between cells and UEs, determines UE-to-cell associations and computes the path gain matrix based on the selected path loss model.

    Notes
    -----
    Path loss calculation does not account for shadowing. Path loss models supported include: 'free-space', 'RMa-LOS', 'RMa-NLOS', 'UMa-LOS', and 'UMa-NLOS'.
    """
    # Warning about height limitations in pathloss models
    if s.params.pathloss_model_name in ['RMa-LOS', 'RMa-NLOS', 'UMa-LOS', 'UMa-NLOS']:
      print(color_text.Yellow(f"\n⚠️  WARNING: Pathloss model '{s.params.pathloss_model_name}' assumes uniform heights (h_BS={s.params.h_BS}m, h_UT={s.params.h_UT}m) for all base stations and UEs. \n\t    Manual modifications to z-coordinates will not propagate to pathloss calculations and may result in inaccurate pathgain values.\n"))
    
    s.distance_matrix=np.linalg.norm(s.cells[:,np.newaxis,:]-s.ues[np.newaxis,:,:],axis=2)
    s.ue_to_cell=s.distance_matrix.argmin(axis=0)
    if s.params.pathloss_model_name=='free-space':
      s.pathloss_dB=lambda x0,x1: 20.0*(np.log10(np.linalg.norm(x1-x0))+np.log10(1e9*s.params.fc_GHz))-147.55 
      pathloss_matrix_dB=20.0*(np.log10(s.distance_matrix)+np.log10(1e9*s.params.fc_GHz))-147.55
      s.pathgain_matrix=np.power(10.0,-pathloss_matrix_dB/10.0)
    elif s.params.pathloss_model_name=='RMa-LOS':
      s.pathloss_model=RMa_pathloss(fc_GHz=s.params.fc_GHz, h_UT=s.params.h_UT, h_BS=s.params.h_BS, LOS=True)
      s.pathgain_matrix=s.pathloss_model(s.cells, s.ues, get_pathgain=True)
    elif s.params.pathloss_model_name=='RMa-NLOS':
      s.pathloss_model=RMa_pathloss(fc_GHz=s.params.fc_GHz, h_UT=s.params.h_UT, h_BS=s.params.h_BS, LOS=False)
      s.pathgain_matrix=s.pathloss_model(s.cells, s.ues, get_pathgain=True)
    elif s.params.pathloss_model_name=='UMa-LOS':
      s.pathloss_model=UMa_pathloss(fc_GHz=s.params.fc_GHz, h_UT=s.params.h_UT, h_BS=s.params.h_BS, LOS=True)
      s.pathgain_matrix=s.pathloss_model(s.cells, s.ues, get_pathgain=True)
    elif s.params.pathloss_model_name=='UMa-NLOS':
      s.pathloss_model=UMa_pathloss(fc_GHz=s.params.fc_GHz, h_UT=s.params.h_UT, h_BS=s.params.h_BS, LOS=False)
      s.pathgain_matrix=s.pathloss_model(s.cells, s.ues, get_pathgain=True)
    else:
      print(color_text.Red(f'pathloss_model_name={s.params.pathloss_model_name} not implemented, quitting 😡'))
      exit(1)

  def get_nues(s):
    """
    Returns the number of UEs in the given object.
    """
    return s.ues.shape[0]

  def get_pathloss_dB(s, d):
    """
    Calculates the path loss in dB between a cell and a user equipment (UE) at a given distance.

    Parameters
    ----------
    d : float
      Distance between the cell and the UE in meters.

    Returns
    -------
    float
      Path loss in dB.

    Raises
    ------
    SystemExit
      If `d` is not a float.
    """
    if isinstance(d, float):
      xyz_cell = np.zeros((1,3))
      xyz_ue   = np.array([[d,   0.0, 0.0]])
      return s.pathloss_model(xyz_cell, xyz_ue).squeeze()
    else:
      print(color_text.Red(f'd={d} is not a float, quitting 😡'))
      exit(1)

  def generate_new_ue_locations(s,seed=0):
    """
    Generate new user equipment (UE) locations and update related attributes.

    Parameters
    ----------
    seed : int, optional
      Random seed for reproducibility (default is 0).

    Notes
    -----
    Updates the UE layout, number of UEs, and initializes related matrices.
    """
    s.erml.generate_new_UE_layout(seed)
    s.ues=s.erml.get_ues()
    s.nues_actual=s.ues.shape[0]
    s.ues=np.column_stack((s.ues, np.full(s.n_ues_actual, s.params.h_UT)))
    s.init_matrices()
    
  def _dbg_get_RMa_pathloss_dB(s):
    """
    Compute the RMa pathloss in dB between cells and UEs.

    Returns
    -------
    numpy.ndarray
      The pathloss values in dB, squeezed to remove single-dimensional entries.

    Raises
    ------
    SystemExit
      If the pathloss_model is not set.
    """
    if s.pathloss_model is None:
      print(color_text.Red(f'pathloss_model is not set to RMa model, quitting 😡'))
      exit(1)
    else: 
      xyz_cells = s.cells
      xyz_ues = s.ues
      return s.pathloss_model(xyz_cells, xyz_ues).squeeze()
    
  def get_full_username(s): 
    """
    Returns the full username of the current user.
    This is intended to automatically populate the 'author' field in plotting methods.

    Returns
    -------
    str
      The full username of the current user, with trailing commas removed.
    """
    return getpwuid(getuid())[4].strip(',')

  def plot(s,grid=False,title='',fnbase='img/CellularReferenceModel',show_attachment=False,show_plot=False,show_voronoi=False,padding_factor=1.02,show_kilometres=False,show_system_rmax_circle=True,show_UE_radius_circle=True,show_pathloss_circles=False,cell_image=None,UE_image=None,cell_image_zoom=5e-2,UE_image_zoom=8e-2, return_figure=False, fmt=['png','pdf'], dbg=False):
    """
    Plot the cellular reference model layout with various optional overlays.

    Parameters
    ----------
    grid : bool, optional
      Whether to display grid lines on the plot.
    title : str, optional
      Title for the plot.
    fnbase : str, optional
      Base filename for saving the plot.
    show_attachment : bool, optional
      If True, show UE-cell attachment lines.
    show_plot : bool, optional
      If True, display the plot interactively.
    show_voronoi : bool, optional
      If True, overlay Voronoi tessellation of cell locations.
    padding_factor : float, optional
      Factor to expand plot limits beyond system radius.
    show_kilometres : bool, optional
      If True, display axes in kilometres.
    show_system_rmax_circle : bool, optional
      If True, draw a circle at the system maximum radius.
    show_UE_radius_circle : bool, optional
      If True, draw a circle at the UE radius.
    show_pathloss_circles : bool, optional
      If True, overlay circles for specific pathloss values.
    cell_image : array-like or None, optional
      Image to use for cell markers.
    UE_image : array-like or None, optional
      Image to use for UE markers.
    cell_image_zoom : float, optional
      Zoom factor for cell images.
    UE_image_zoom : float, optional
      Zoom factor for UE images.
    return_figure : bool, optional
      If True, return the plot object instead of saving.
    fmt : list of str, optional
      List of file formats for saving the plot.
    dbg : bool, optional
      If True, print debug information.

    Returns
    -------
    plot : Plot_CRM_layout or None
      The plot object if `return_figure` is True, otherwise None.
    """
    plot_axlim = (-padding_factor*s.erml.system_rmax, 
                   padding_factor*s.erml.system_rmax,)
    plot=Plot_CRM_layout(xlim=plot_axlim,ylim=plot_axlim,grid=grid,cell_image=cell_image,UE_image=UE_image,cell_image_zoom=cell_image_zoom,UE_image_zoom=UE_image_zoom,fnbase=fnbase)
    plot.base(s.cells,s.ues, show_kilometres=show_kilometres)
    if show_voronoi: plot.voronoi(s.cells[:,:2])
    if show_attachment: plot.distance(cells=s.cells,ues=s.ues,plot_type='nearest')
    if s.params.author=='': s.params.author=s.get_full_username()
    if show_UE_radius_circle:
      UE_radius_circle=plt.Circle((0.0,0.0),s.erml.UE_radius,color='grey',fill=False,lw=2,linestyle='dashed',zorder=8)
      plot.ax.add_patch(UE_radius_circle)
    if show_system_rmax_circle: # Keith Briggs 2024-08-29
        circle=plt.Circle((0.0,0.0),s.erml.system_rmax,color='black',fill=False,lw=1,linestyle='dotted',zorder=8)
        plot.ax.add_patch(circle)
    # which circles to draw...
    if 'NLOS' in s.params.pathloss_model_name:
      pathloss_circles_dB=(140.0,160.0,170.0)
    else:
      pathloss_circles_dB=(100.0,110.0,120.0,)
    if show_pathloss_circles: # Keith Briggs 2024-08-16
      if s.params.pathloss_model_name=='free-space':
        pathloss_dB=lambda d: 20.0*(np.log10(d)+np.log10(1e9*s.params.fc_GHz))-147.55 
      elif 'RMa' in s.params.pathloss_model_name:
        pathloss_dB=lambda d: s.get_pathloss_dB(d)
      elif 'UMa' in s.params.pathloss_model_name:
        pathloss_dB=lambda d: s.get_pathloss_dB(d)
      else:
        print(color_text.Red(f'{s.params.pathloss_model_name} not implemented!'))
        return
      for pl_dB in pathloss_circles_dB:
        f=lambda x: pathloss_dB(x)-pl_dB
        root=root_scalar(f,bracket=(10.0,1e4))
        d=root.root
        if dbg: print(color_text.IYellow(f'{s.params.pathloss_model_name}: root_scalar finds d={d:g} for pathloss={pl_dB:.0f}dB'))
        circle=plt.Circle((0.0,0.0),d,color='purple',fill=False)
        plot.ax.add_patch(circle)
        theta=0.25*np.pi
        xy=(d*np.cos(theta),d*np.sin(theta))
        plot.ax.annotate(f'{pl_dB:.0f}dB',xy,color='purple',fontsize=8)
    if show_plot: plt.show()
    if title: plot.ax.set_title(title)
    plot.fig.tight_layout()
    if return_figure: return plot
    plot.savefig(timestamp=True,fmt=fmt, author=s.params.author)
  # END CellularReferenceModel class

class Plot_CRM_layout:
  """
  Class for visualizing cellular reference model layouts, including cells, user equipment (UE), distances, and Voronoi diagrams.

  Parameters
  ----------
  fnbase : str, optional
    Base filename for saving figures (default is '').
  xlim : tuple of float, optional
    X-axis limits for the plot (default is (-4.5, 4.5)).
  ylim : tuple of float, optional
    Y-axis limits for the plot (default is (-4.5, 4.5)).
  grid : bool, optional
    Whether to display a grid on the plot (default is False).
  cell_image : str or None, optional
    Path to image file for representing cells (default is None).
  UE_image : str or None, optional
    Path to image file for representing UEs (default is None).
  cell_image_zoom : float, optional
    Zoom factor for cell images (default is 5e-2).
  UE_image_zoom : float, optional
    Zoom factor for UE images (default is 8e-2).

  Attributes
  ----------
  fig : matplotlib.figure.Figure
    The matplotlib figure object.
  ax : matplotlib.axes.Axes
    The axes object for plotting.
  cell_image : str or None
    Path to cell image file.
  cell_image_zoom : float
    Zoom factor for cell images.
  UE_image : str or None
    Path to UE image file.
  UE_image_zoom : float
    Zoom factor for UE images.
  fnbase : str
    Base filename for saving figures.
  """
  def __init__(self,fnbase='',xlim=(-4.5,4.5),ylim=(-4.5,4.5),grid=False,cell_image=None,UE_image=None,cell_image_zoom=5e-2,UE_image_zoom=8e-2):
    self.fig=plt.figure(figsize=(6,6))
    self.ax=self.fig.add_subplot()
    if grid: self.ax.grid(color='gray',lw=0.5,alpha=0.5)
    self.ax.set_aspect('equal')
    self.ax.set_xlim(*xlim)
    self.ax.set_ylim(*ylim)
    self.fig.tight_layout()
    self.cell_image=cell_image
    self.cell_image_zoom=cell_image_zoom
    self.UE_image=UE_image
    self.UE_image_zoom=UE_image_zoom
    if fnbase: self.fnbase=fnbase
    else: self.fnbase='img/energy_reference_model_layout_generator_test'

  def getImage(self,path,zoom=1):
    # https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points?
    return OffsetImage(plt.imread(path),zoom=zoom)

  def base(self,cells,ues,show_kilometres=False):
    """
    Plots the base layout of cells and user equipments (UEs) on the current axes.

    Parameters
    ----------
    cells : np.ndarray
      Array of cell coordinates with shape (N, 2).
    ues : np.ndarray
      Array of UE coordinates with shape (M, 2).
    show_kilometres : bool, optional
      If True, axis labels and ticks are shown in kilometres. Default is False.

    Notes
    -----
    If image files for cells or UEs are specified and can be loaded, they are used as markers; otherwise, default scatter markers are used.
    """
    if self.cell_image is None:
      self.ax.scatter(cells[:,0],cells[:,1],marker='o',s=50,color='red',alpha=1.0,zorder=3)
    else:
      image_ok=True
      try:
        cell_image=self.getImage(self.cell_image,zoom=self.cell_image_zoom)
      except:
        print(color_text.Red(f'Could not open image file {self.cell_image}!'))
        image_ok=False
      if image_ok:
        for x,y in zip(cells[:,0],cells[:,1]):
          self.ax.add_artist(AnnotationBbox(cell_image,(x,y),frameon=False))
    if self.UE_image is None:
      self.ax.scatter(ues[:,0],ues[:,1],marker='.',s=30,color='blue',zorder=3)
    else:
      image_ok=True
      try:
        UE_image=self.getImage(self.UE_image,zoom=self.UE_image_zoom)
      except:
        print(color_text.Red(f'Could not open image file {self.UE_image}!'))
        image_ok=False
      if image_ok:
        for x,y in zip(ues[:,0],ues[:,1]):
          self.ax.add_artist(AnnotationBbox(UE_image,(x,y),frameon=False))
    if show_kilometres:
      self.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1000:.1f}'))
      self.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{y/1000:.1f}'))
      self.ax.set_xlabel('km')
      self.ax.set_ylabel('km')

  def distance(self,cells=None,ues=None,plot_type='all',label_distance=False):
    """
    Plots distances between cells and user equipments (UEs) on the current axes.

    Parameters
    ----------
    cells : array-like, optional
      Coordinates of the cells. If None, uses the first scatter plot data from `self.ax`.
    ues : array-like, optional
      Coordinates of the UEs. If None, uses the second scatter plot data from `self.ax`.
    plot_type : {'all', 'actual_attachment', 'nearest'}, default='all'
      Type of distance visualization:
      - 'all': Draws lines between every cell and every UE.
      - 'actual_attachment': (Not implemented) Intended for actual associations.
      - 'nearest': Draws lines between each UE and its nearest cell.
    label_distance : bool, default=False
      If True, annotates the plotted lines with the computed distances.

    Returns
    -------
    None
    """
    if cells is None: cells=self.ax.collections[0].get_offsets()
    if ues is None:
      ues=self.ax.collections[1].get_offsets()
    if plot_type=='all': # draw a line between each cell and each ue
      for cell,ue in product(cells,ues):
        self.ax.plot([cell[0],ue[0]],[cell[1],ue[1]],color='black',lw=0.5,alpha=0.5,zorder=1)
        dist=np.linalg.norm(cell-ue)
        self.ax.text((cell[0]+ue[0])/2,(cell[1]+ue[1])/2,f'{dist:.2f}',fontsize=6,color='black',alpha=0.7,zorder=1)
    if plot_type=='actual_attachment':
      print('Not implemented yet!')
      # FIXME: This needs the actual association matrix / vector from the CRM_SA.
    if plot_type=='nearest':
      distances=np.linalg.norm(cells[:,np.newaxis,:]-ues[np.newaxis,:,:],axis=2)
      nearest_cell_index=np.argmin(distances,axis=0)
      for i,ue in enumerate(ues):
        self.ax.plot([cells[nearest_cell_index[i],0],ue[0]],
                     [cells[nearest_cell_index[i],1],ue[1]],
                      color='gray',lw=0.5,alpha=0.5)
        if label_distance:
          d=np.linalg.norm(cells[nearest_cell_index[i]]-ue)
          self.ax.text((cells[nearest_cell_index[i],0]+ue[0])/2,(cells[nearest_cell_index[i],1]+ue[1])/2,f'{d:.2f}',fontsize=6,color='black',alpha=0.7)
  
  def voronoi(self, cells=None):
    """
    Plots the Voronoi diagram for a given set of 2D cell coordinates.

    Parameters
    ----------
    cells : np.ndarray, optional
      A 2D numpy array of shape (n, 2) representing the coordinates of the cells.
      If None, raises a ValueError.

    Raises
    ------
    ValueError
      If `cells` is not a non-empty 2D numpy array with shape (n, 2).

    Notes
    -----
    The axis limits are preserved after plotting the Voronoi diagram.
    """
    if cells is None or not isinstance(cells, np.ndarray) or cells.ndim != 2 or cells.shape[1] != 2:
        raise ValueError("cells must be a non-empty 2D numpy array with shape (n, 2)")
    try:
        vor = Voronoi(cells)
    except Exception as e:
        print(f"Error computing Voronoi diagram: {e}")
        return
    xlim = self.ax.get_xlim()
    ylim = self.ax.get_ylim()
    voronoi_plot_2d(vor, ax=self.ax, show_vertices=False, show_points=False, 
                    line_colors='green', line_width=1.0, line_alpha=0.5)
    self.ax.set_xlim(xlim)
    self.ax.set_ylim(ylim)

  def savefig(self,fn=None,timestamp=True,fmt=['png','pdf'], author=''):
    """
    Save the current figure to file(s) in specified formats, optionally adding a timestamp and author.

    Parameters
    ----------
    fn : str or None, optional
      Base filename for saving the figure. If None, uses the existing filename base.
    timestamp : bool, default=True
      Whether to add a timestamp and author annotation to the figure.
    fmt : list of str, default=['png', 'pdf']
      List of file formats to save the figure in.
    author : str, optional
      Name of the author to include in the timestamp annotation.

    Notes
    -----
    Prints commands to open the saved files depending on the operating system.
    """
    if fn is not None: self.fnbase=fn
    if timestamp:
      fig_timestamp(self.fig,fontsize=6,color='black',alpha=0.7,
                    rotation=0,prespace='  ',author=author)
    commands={'Darwin':'open','Linux':'eog','Windows':''}
    system=platform_system()
    command=commands.get(system,'')
    for ext in fmt:
      self.fig.savefig(f'{self.fnbase}.{ext}',pad_inches=0.1)
      if command:
        if system=='Linux':
          if ext=='png': print(color_text.IYellow(f'eog {self.fnbase}.{ext} &'))
          if ext=='pdf': print(color_text.IYellow(f'evince {self.fnbase}.{ext} &'))
        else:
          print(color_text.Yellow(f'{command} {self.fnbase}.{ext}'))
      else:
        print(color_text.Yellow(f'{self.fnbase}.{ext}'))

if __name__=='__main__':
  np.set_printoptions(precision=4,linewidth=200,suppress=False)

  # Example: CellularReferenceModel RMa LOS
  crm=CellularReferenceModel(params=CRMParameters(n_cells=7,n_ues=30,pathloss_model_name='RMa-LOS',cell_radius_m=5e3,UE_layout='uniform',UE_radius=1e4,verbose=True))
  G0=crm.pathgain_matrix
  crm.plot(show_plot=False,show_voronoi=True,show_attachment=True,show_kilometres=True,show_pathloss_circles=True,fnbase='./CellularReferenceModel_RMa-LOS',)

  # Example: CellularReferenceModel UMa LOS
  crm=CellularReferenceModel(params=CRMParameters(n_cells=7,n_ues=30,pathloss_model_name='UMa-LOS',cell_radius_m=5e3,UE_layout='uniform',UE_radius=1e4,verbose=True))
  G0=crm.pathgain_matrix
  crm.plot(show_plot=False,show_voronoi=True,show_attachment=True,show_kilometres=True,show_pathloss_circles=True,fnbase='./CellularReferenceModel_UMa-LOS',)

  # Example: CellularReferenceModel RMa NLOS
  crm=CellularReferenceModel(params=CRMParameters(n_cells=19,n_ues=100, pathloss_model_name='RMa-NLOS', cell_radius_m=5e3))
  G0=crm.pathgain_matrix
  crm.plot(show_plot=False, show_voronoi=True, show_attachment=True, show_kilometres=True, fnbase='./CellularReferenceModel_RMa-NLOS', show_pathloss_circles=True)
