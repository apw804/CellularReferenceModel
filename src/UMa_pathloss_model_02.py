"""
Changelog
---------
- 2025-05-06 (v02): Updated docstrings. [Author: Kishan Sthankiya]
- 2024-10-18 (v01): Vectorized pathloss. [Author: Kishan Sthankiya]
- 2022-03-07: Implemented NLOS case. [Author: Keith Briggs]
- 2021-10-29: 
      - Modified pathloss function to take two arguments (cell position and UE position),
        allowing pathloss to depend on absolute position, not just distance. [Author: Keith Briggs]

- 2021-05-17: Initial implementation. [Author: Keith Briggs]
- 2020-10-05: Referenced from ~/Voronoi-2.1/try_3GPP_UMa_pathloss_01.py. [Author: Keith Briggs]

Notes
-----
- To perform a self-test (generates a plot), run:
  `python3 UMa_pathloss_model_02.py`
"""

import numpy as np

# Add the src folder to the system path to enable imports when running from a different directory
import sys
sys.path.append('../src')

class UMa_pathloss:
  """
  Urban macrocell dual-slope pathloss model, from 3GPP standard 36.873,
  Table 7.2-1.

  This model covers the cases 3D-UMa LOS and NLOS:
  - 3D-UMa: Three-dimensional urban macrocell model.
  - LOS   : Line-of-sight.
  - NLOS  : Non-line-of-sight.

  References:
  - 3GPP standard 36.873: https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=2574

  Parameters
  ----------
  fc_GHz : float, optional
    Centre frequency in GigaHertz (default is 3.5 GHz).
  h_UT : float, optional
    Height of the User Terminal (UE) in metres (default is 2.0 m).
    Must be between 1.5 m and 22.5 m.
  h_BS : float, optional
    Height of the Base Station in metres (default is 25.0 m).
    Must be less than or equal to 25.0 m.
  LOS : bool, optional
    Whether the line-of-sight model is to be used (default is True).
  AIMM_simulator : bool, optional
    Whether the AIMM simulator is being used (default is False).

  Attributes
  ----------
  fc_GHz : float
    Centre frequency in GHz.
  log10fc : float
    Logarithm (base 10) of the centre frequency.
  LOS : bool
    Indicates if the LOS model is used.
  h_UT : float
    Height of the User Terminal (UE) in metres.
  h_BS : float
    Height of the Base Station in metres.
  dBP_prime : float
    Breakpoint distance in metres.
  d2D_m : ndarray or None
    2D distance between cells and UEs (computed during call).
  d3D_m : ndarray or None
    3D distance between cells and UEs (computed during call).


  Notes
  -----
  - The model assumes specific conditions for urban macrocells and may not be applicable to other scenarios.
  - The LOS probability is not implemented and should be handled separately.
  """

  def __init__(s, fc_GHz=3.5, h_UT=2.0, h_BS=25.0, LOS=True, AIMM_simulator=False):
    """
    Initialize a pathloss model instance.

    Raises
    ------
    ValueError
      If `h_UT` is not between 1.5 m and 22.5 m.
      If `h_BS` is greater than 25.0 m.
    """
    s.fc_GHz = fc_GHz
    s.log10fc = np.log10(s.fc_GHz)
    s.LOS = LOS
    if np.less_equal(1.5, h_UT) and np.less(h_UT, 22.5):
      s.h_UT = h_UT
    else:
      raise ValueError('h_UT must be between 1.5 m and 22.5 m')
    if np.less_equal(h_BS, 25.0):
      s.h_BS = h_BS
    else:
      raise ValueError('h_BS must be less than or equal to 25.0 m')

    s.distance_normalisation_value = 1.0
    s.fc_norm_value = 1.0
    s.c = 3e8
    s.h_E = 1.0
    s.h_BS_prime = h_BS - s.h_E
    s.h_UT_prime = h_UT - s.h_E
    s.dBP_prime = 4 * s.h_BS_prime * s.h_UT_prime * (s.fc_GHz / s.c)

    # LOS PL1
    s.los1_t1 = 28.0
    s.los1_t3 = 20.0 * s.log10fc
    s.los1_consts = s.los1_t1 + s.los1_t3

    # LOS PL2
    s.los2_t1 = 28.0
    s.los2_t3 = 20 * s.log10fc
    s.los2_t4 = (-9) * np.log10(np.power(s.dBP_prime, 2) + np.power((s.h_BS - s.h_UT), 2))
    s.los2_consts = s.los2_t1 + s.los2_t3 + s.los2_t4

    # NLOS
    s.nlos_consts = 13.54 + 20 * s.log10fc - (0.6 * (s.h_UT - 1.5))

    s.d2D_m = None
    s.d3D_m = None
    s.AIMM_simulator = AIMM_simulator

  def __call__(s, xyz_cells, xyz_UEs, return_LOS_PL1=False, return_LOS_PL2=False, 
         print_coeffs=False, get_pathgain=False):
    """
    Compute the pathloss or pathgain between cells and UEs.

    Parameters
    ----------
    xyz_cells : ndarray
      Array of shape (M, 3) representing the 3D positions of M cells.
    xyz_UEs : ndarray
      Array of shape (N, 3) representing the 3D positions of N UEs.
    return_LOS_PL1 : bool, optional
      If True, return the LOS PL1 pathloss values (default is False).
    return_LOS_PL2 : bool, optional
      If True, return the LOS PL2 pathloss values (default is False).
    print_coeffs : bool, optional
      If True, print the coefficients used in the pathloss calculation
      (default is False).
    get_pathgain : bool, optional
      If True, return the pathgain instead of the pathloss (default is False).

    Returns
    -------
    ndarray
      Pathloss or pathgain values between each cell and UE. The shape of
      the output is (M, N), where M is the number of cells and N is the
      number of UEs.

    Notes
    -----
    - The LOS probability is not implemented and should be handled separately.
    - The distances, building heights, and other parameters are not checked
      to ensure that this pathloss model is applicable.
    """
    if s.AIMM_simulator:
      if xyz_cells.ndim == 1:
        xyz_cells = xyz_cells[np.newaxis, :]
      if xyz_UEs.ndim == 1:
        xyz_UEs = xyz_UEs[np.newaxis, :]
    s.d2D_m = np.linalg.norm(xyz_cells[:, np.newaxis, :2] - xyz_UEs[np.newaxis, :, :2], axis=2)
    s.d3D_m = np.linalg.norm(xyz_cells[:, np.newaxis, :] - xyz_UEs[np.newaxis, :, :], axis=2)
    PL3D_UMa_LOS_PL1 = s.los1_consts + (22.0 * np.log10(s.d3D_m))
    if return_LOS_PL1:
      return PL3D_UMa_LOS_PL1
    PL3D_UMa_LOS_PL2 = s.los2_consts + (40.0 * np.log10(s.d3D_m))
    if return_LOS_PL2:
      return PL3D_UMa_LOS_PL2
    PL3D_UMa_LOS = np.where(np.less(s.d2D_m, s.dBP_prime), PL3D_UMa_LOS_PL1, PL3D_UMa_LOS_PL2)
    if s.LOS:
      if get_pathgain:
        return 1.0 / np.power(10.0, PL3D_UMa_LOS / 10.0)
      if s.AIMM_simulator:
        return PL3D_UMa_LOS[0]
      return PL3D_UMa_LOS
    else:
      PL3D_UMa_NLOS = s.nlos_consts + 39.08 * np.log10(s.d3D_m)
      if get_pathgain:
        return 1.0 / np.power(10.0, np.maximum(PL3D_UMa_LOS, PL3D_UMa_NLOS) / 10.0)
      if s.AIMM_simulator:
        return np.maximum(PL3D_UMa_LOS, PL3D_UMa_NLOS)[0]
      return np.maximum(PL3D_UMa_LOS, PL3D_UMa_NLOS)
# END UMa_pathloss
 
def test_UMa_pathloss_00():
  """
  Test the UMa_pathloss function for calculating 3D Urban Macro (UMa) pathloss.

  This function tests the UMa_pathloss model by providing example inputs for 
  base station (BS) and user equipment (UE) coordinates, and prints the 
  resulting pathloss values. The test assumes a non-line-of-sight (NLOS) 
  scenario.

  Parameters
  ----------
  None

  Returns
  -------
  None
    This function does not return any value. It prints the pathloss values 
    and their shape to the console.

  Notes
  -----
  - The `xyz_cells` array represents the coordinates of the base stations 
    (BS) in a 3D space, with shape (M, 3), where M is the number of cells.
  - The `xyz_ues` array represents the coordinates of the user equipment (UE) 
    in a 3D space, with shape (N, 3), where N is the number of UEs.
  - The `UMa_pathloss` function is expected to compute the pathloss between 
    each BS and each UE, returning a 2D array of shape (M, N) containing the 
    pathloss values in dB.
  - The test uses a carrier frequency (`fc_GHz`) of 3.5 GHz, a UE height 
    (`h_UT`) of 1.5 meters, and a BS height (`h_BS`) of 25.0 meters.
  - The test assumes a non-line-of-sight (NLOS) propagation condition.

  Example
  -------
  >>> test_UMa_pathloss_00()
  PL3D_UMa_LOS=
  [[...]]; 	Shape=(2, 3)
  """
  xyz_cells = np.array([[0.0, 0.0, 25.0],[8.0, 5.0, 25.0]])
  xyz_ues = np.array([[10.0, 10.0, 1.5], [2.0, 2.0, 1.5], [5.0, 5.0, 1.5]])
  PL = UMa_pathloss(fc_GHz=3.5, h_UT=1.5, h_BS=25.0, LOS=False)
  a=PL(xyz_cells, xyz_ues)
  print(f'PL3D_UMa_LOS=\n{a}; \nShape={a.shape}\n') 

def test_UMa_pathloss_01():
  """
  Test the UMa (Urban Macro) path loss model with a specific configuration.

  This function evaluates the UMa path loss model for a single base station 
  (BS) and a range of user equipment (UE) positions. The BS is fixed at a 
  specific location, while the UEs vary along the x-axis. The function 
  calculates the path loss for a non-line-of-sight (NLOS) scenario.

  Parameters
  ----------
  None

  Returns
  -------
  None
    This function does not return any value. It prints the calculated 
    path loss values and their shape.

  Notes
  -----
  - The base station (BS) is located at (0.0, 0.0, 25.0) meters.
  - The user equipment (UE) positions vary along the x-axis from 10.0 to 
    5000.0 meters, with a fixed height of 1.5 meters.
  - The carrier frequency is set to 3.5 GHz.
  - The path loss is calculated for a non-line-of-sight (NLOS) scenario.

  Examples
  --------
  >>> test_UMa_pathloss_01()
  PL3D_UMa_LOS=
  [array of path loss values]; 	Shape=(1, 1000)
  """
  xyz_cells = np.array([[0.0, 0.0, 25.0]]) 
  x = np.linspace(10.0, 5000.0, 1000)
  xyz_ues = np.column_stack((x, np.zeros_like(x), np.full_like(x, 1.5)))
  PL = UMa_pathloss(fc_GHz=3.5, h_UT=1.5, h_BS=25.0, LOS=False)
  a=PL(xyz_cells, xyz_ues)
  print(f'PL3D_UMa_LOS=\n{a}; \nShape={a.shape}\n')

def plot_UMa_pathloss_or_pathgain(plot_type='pathloss', fc_GHz=3.5, h_UT=2.0, h_BS=25.0, zoom_box=False, print_10m_pl=False, author=' ', x_min=10.0, x_max=1000.0):
  """
  Plot the 3GPP UMa pathloss or pathgain model predictions as a self-test.

  This function generates a plot of the 3GPP UMa pathloss or pathgain models 
  for both LOS (Line-of-Sight) and NLOS (Non-Line-of-Sight) scenarios. It 
  also includes a free-space pathloss reference and an optional zoomed-in 
  view of the plot.

  Parameters
  ----------
  plot_type : str, optional
    Type of plot to generate. Options are:
    - 'pathloss': Plot the pathloss in dB (default).
    - 'pathgain': Plot the pathgain.
  fc_GHz : float, optional
    Carrier frequency in GHz (default is 3.5 GHz).
  h_UT : float, optional
    Height of the User Terminal (UE) in meters (default is 2.0 m).
  h_BS : float, optional
    Height of the Base Station (BS) in meters (default is 25.0 m).
  zoom_box : bool, optional
    If True, include a zoomed-in view of the plot (default is False).
  print_10m_pl : bool, optional
    If True, print the pathloss values at 10 meters for LOS, NLOS, and 
    free-space scenarios (default is False).
  author : str, optional
    Author name to include in the plot timestamp (default is an empty string).
  x_min : float, optional
    Minimum x-axis value for the plot, representing the minimum distance 
    in meters (default is 10.0 m).
  x_max : float, optional
    Maximum x-axis value for the plot, representing the maximum distance 
    in meters (default is 1000.0 m).

  Raises
  ------
  ImportError
    If required modules (e.g., matplotlib) are not installed.

  Notes
  -----
  - The function uses the `UMa_pathloss` class to compute pathloss and 
    pathgain values for LOS and NLOS scenarios.
  - The free-space pathloss is included as a reference.
  - The zoomed-in view highlights a specific region of the plot for better 
    visualization of details.

  Examples
  --------
  Plot the pathloss model predictions with default parameters:
  >>> plot_UMa_pathloss_or_pathgain()

  Plot the pathgain model predictions with a zoomed-in view:
  >>> plot_UMa_pathloss_or_pathgain(plot_type='pathgain', zoom_box=True)

  Print the pathloss values at 10 meters:
  >>> plot_UMa_pathloss_or_pathgain(print_10m_pl=True)

  """
  try:
    import matplotlib.pyplot as plt
    from fig_timestamp_01 import fig_timestamp
  except ImportError as e:
    print(f"Error importing modules: {e}")
    raise

  # Setup figure
  fig, ax = plt.subplots(figsize=(8, 6))
  ax.grid(color='gray', alpha=0.5, lw=0.5)
  
  # Define coordinates for cells and UEs
  xyz_cells = np.array([[0.0, 0.0, h_BS]])
  x = np.linspace(x_min, x_max, 4990)
  xyz_ues = np.column_stack((x, np.zeros_like(x), np.full_like(x, h_UT)))

  # Plot NLOS
  PL = UMa_pathloss(fc_GHz=fc_GHz, h_UT=h_UT, h_BS=h_BS, LOS=False)
  dBP_NLOS = PL.dBP_prime
  dBP_NLOS_index = np.searchsorted(x, dBP_NLOS)
  PL_NLOS_dB = PL(xyz_cells, xyz_ues).squeeze()
  PG_NLOS = 1.0 / np.power(10.0, PL_NLOS_dB / 10.0)
  
  if plot_type == 'pathloss':
    ax.set_title(f'3GPP UMa pathloss models (dBP={dBP_NLOS:.0f}m)')
    ax.set_ylabel('pathloss (dB)')
    line=ax.plot(x, PL_NLOS_dB, lw=2, label=r'NLOS ($\sigma=8$)', color='blue')
    line_color = line[0].get_color()
    ax.vlines(dBP_NLOS, 0, PL_NLOS_dB[dBP_NLOS_index], line_color, 'dotted', lw=2)
    ax.fill_between(x, PL_NLOS_dB - 6, PL_NLOS_dB + 6, color=line_color, alpha=0.2)
    ax.set_ylim(50)
  else:
    ax.set_title(f'3GPP UMa pathgain models (dBP={dBP_NLOS:.0f}m)')
    ax.set_ylabel('pathgain')
    ax.plot(x, PG_NLOS, lw=2, label='NLOS pathgain')
  
  # Plot LOS
  PL = UMa_pathloss(fc_GHz=fc_GHz, h_UT=h_UT, h_BS=h_BS, LOS=True)
  dBP_LOS = PL.dBP_prime
  dBP_LOS_index = np.searchsorted(x, dBP_LOS)
  PL_LOS_dB = PL(xyz_cells, xyz_ues).squeeze()
  PG_LOS = 1.0 / np.power(10.0, PL_LOS_dB / 10.0)
  
  if plot_type == 'pathloss':
    line=ax.plot(x, PL_LOS_dB, lw=2, label=r'LOS ($\sigma=4$)', color='orange')
    line_color = line[0].get_color()
    sigma = np.where(np.less_equal(x, dBP_LOS), 4.0, 4.0)
    ax.vlines(dBP_LOS, 0, PL_LOS_dB[dBP_LOS_index], line_color, 'dotted', lw=2)
    ax.fill_between(x, PL_LOS_dB - sigma, PL_LOS_dB + sigma, color=line_color, alpha=0.2)
    ax.set_xlim(0, np.max(x))
    fnbase = 'UMa_pathloss_model_02' # FIXME: This should go in the parent img folder
  else:
    ax.plot(x, PG_LOS, lw=2, label='LOS pathgain')
    ax.set_ylim(0)
    ax.set_xlim(0, 1000)
    fnbase = 'UMa_pathgain_model_02' # FIXME: This should go in the parent img folder

  # Plot the Free-space pathloss as a reference
  fs_pathloss_dB = (20 * np.log10(PL.d3D_m) + 20 * np.log10(fc_GHz*1e9) - 147.55).squeeze(axis=0)
  fs_pathloss = np.power(10.0, fs_pathloss_dB / 10.0)
  fs_pathgain = 1.0 / fs_pathloss
  if plot_type == 'pathloss':
    fspl_line = ax.plot(x, fs_pathloss_dB, lw=2, label='Free-space pathloss', color='red')
  else:
    fspl_line = ax.plot(x, fs_pathgain, lw=2, label='Free-space pathloss', color='red')

  # Add zoom box at lower left of plot
  if zoom_box and plot_type == 'pathloss':

    # Define the area you want to zoom in on
    x1,x2,y1,y2 = 0.0, 50.1, 70.0, 80.0

    # Define where you want the zoom box to be placed
    axins = ax.inset_axes([0.50, 0.1, 0.2, 0.33])

    # Plot the zoomed area
    axins.plot(x, PL_NLOS_dB, lw=2, label='NLOS pathloss', color='blue')
    axins.plot(x, PL_LOS_dB, lw=2, label='LOS pathloss', color='orange')
    axins.plot(x, fs_pathloss_dB, lw=2, label='Free-space pathloss', color='red')
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticks(np.arange(x1, x2, 10))
    axins.set_yticks(np.arange(y1, y2, 1))
    axins.tick_params(axis='both', direction='in')
    axins.grid(color='gray', alpha=0.7, lw=0.5)
    ax.indicate_inset_zoom(axins, edgecolor='gray')
  
  # Final plot adjustments
  ax.set_xlabel('distance (metres)')
  ax.legend()
  fig.tight_layout()

  # Print the pathloss at 10 metres
  if print_10m_pl:
    BLUE =   "\033[38;5;027m"
    ORANGE = "\033[38;5;202m"
    RED =    "\033[38;5;196m"
    RESET =  "\033[0m"
    print(f'\nPathloss at 10 metres:')
    print('----------------------')
    print(f'{BLUE}UMa-NLOS:       {PL_NLOS_dB[0]:.2f} dB')
    print(f'{ORANGE}UMa-LOS:        {PL_LOS_dB[0]:.2f} dB')
    print(f'{RED}Free-space:     {fs_pathloss_dB[0]:.2f} dB{RESET}\n')
  
  # Add timestamp and save figures
  fig_timestamp(fig, rotation=0, fontsize=6, author=author)
  fig.savefig(f'{fnbase}.png')
  fig.savefig(f'{fnbase}.pdf')
  print(f'eog {fnbase}.png &')
  print(f'evince {fnbase}.pdf &')


if __name__=='__main__':
  # test_UMa_pathloss_00()
  # test_UMa_pathloss_01()
  plot_UMa_pathloss_or_pathgain(plot_type='pathgain', zoom_box=True, print_10m_pl=True, author='Kishan Sthankiya') # simple self-test

