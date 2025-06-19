"""
Changelog:
----------
- 2025-05-13 (v03): Updated docstrings. [Author: Kishan Sthankiya]
- 2024-08-20:
  - Added an option to the init method to specify if this is being used in the AIMM simulator. 
    If true, the call method checks the shape of the input and returns pathloss values accordingly.
- 2024-08-19:
  - Printed pathloss at 10 metres in the test plot.
  - Added an option for a zoom box at the lower left of the plot.
  - Tweaks by Keith Briggs.

2024-08-07:
  - Introduced new plotting for pathloss and pathgain with FSPL comparison.
"""

import numpy as np

# Add the src folder to the system path to enable imports when running from a different directory
import sys
sys.path.append('../src')

class RMa_pathloss:
  """
  Rural macrocell , from 3GPP standard 38.901 v18.0.0 (Release 18),
  Table 7.4.1-1 pp 27. 

  This model covers the cases 3D-RMa LOS and NLOS:
  - 3D-RMa: Three dimensional rural macrocell model.
  - LOS   : Line-of-sight.
  - NLOS  : Non-line-of-sight.

  References:
   - 3GPP standard 38.901: https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/18.00.00_60/tr_138901v180000p.pdf.

  Parameters
  ----------
  fc_GHz : float, optional
    Centre frequency in GigaHertz (default 3.5 GHz).
  h_UT : float, optional
    Height of User Terminal (UE) in metres  (default 1.5 m).
  h_BS : float, optional
    Height of Base Station in metres (default 35.0 m).
  LOS: bool, optional
    Whether line-of-sight model is to be used (default True).
  h : float, optional
    Average building height (default 5, used in NLOS case only)
  W : float, optional
    Street width (default 20, used in NLOS case only)
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
  c : float
    Speed of light constant.
  h_UT : float
    Height of User Terminal (UE) in metres.
  h_BS : float
    Height of the Base Station in metres.
  avg_building_height : float
    Mean building height in metres.
  avg_street_width : float
    Mean street width in metres.
  distance_normalisation_value : float
    Value to ensure all distances are given in metres.
  fc_norm_value : float
    Value to ensure all frequencies are given in GHz.
  dBP : float
    Breakpoint distance in metres.
  d3D_m : ndarray or None
    3D distance between cells and UEs (computed during call).
  
  Notes
  -----
  - FIXME-1: If the h_BS and/or h_UT deviate from the default values, the pathloss may not be accurate as these are fixed on the init and do not dynamically change. They are both used to calculate the dBP.
  - The model assumes specific conditions for rural macrocells and may not be applicable to other scenarios.
  - The LOS probability is not implemented and should be handled separately.
  """
  def __init__(s,fc_GHz=3.5,h_UT=1.5,h_BS=35.0,LOS=True,h=5.0,W=20.0, AIMM_simulator=False):
    '''
    Initialize a pathloss model instance.

    '''
    s.fc_GHz=fc_GHz
    s.log10fc=np.log10(s.fc_GHz)
    s.LOS=LOS
    s.c=3e8
    if 1.0 <= h_UT <= 10.0:
      s.h_UT=h_UT
    else:
      raise ValueError('h_UT must be between 1.0m and 10.0m')
    if 10.0 <= h_BS <= 150.0:    
      s.h_BS=h_BS
    else:
      raise ValueError('h_BS must be between 10.0m and 150.0m')
    if 5.0 <= h <= 50.0:
      s.avg_building_height=h
    else:
      raise ValueError('Average building height (h) must be between 5.0m and 50.0m')
    if 5.0 <= W <= 50.0:
      s.avg_street_width=W
    else:
      raise ValueError('Average street width (W) must be between 5.0m and 50.0m')
    
    s.distance_normalisation_value=1.0
    s.fc_norm_value=1.0
    s.dBP = 2 * np.pi * s.h_BS * s.h_UT * s.fc_GHz * 1e9 / s.c  # see Note 5 and Note 6 in Table 7.4.1-1

    # LOS terms
    s.los1_t1 = 40*np.pi * (s.fc_GHz/3.0)
    s.los1_t2 = min(0.03 * s.avg_building_height**1.72, 10.0)
    s.los1_t3 = -min(0.044 * s.avg_building_height**1.72, 14.77)
    s.los1_t4 = 0.002*np.log10(s.avg_building_height)

    # NLOS
    s.nlos_consts= 161.04-7.1*np.log10(s.avg_street_width)+7.5*np.log10(s.avg_building_height)-(24.37 - 3.7*(s.avg_building_height/s.h_BS)**2)*np.log10(s.h_BS)+20*np.log10(s.fc_GHz)-(3.2*np.log10(11.75*s.h_UT)**2 - 4.97)
    s.nlos_t5=(43.42 - 3.1*np.log10(s.h_BS))

    s.d3D_m=None
    s.AIMM_simulator=AIMM_simulator


  def __call__(s, xyz_cells, xyz_UEs, return_PL1=False, return_PL2=False, print_coeffs=False, get_pathgain=False):
    """
    Return the pathloss or pathgain between cells and UEs.

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
    """
    if s.AIMM_simulator:
      if xyz_cells.ndim == 1: 
         xyz_cells = xyz_cells[np.newaxis, :]
      if xyz_UEs.ndim == 1: 
         xyz_UEs = xyz_UEs[np.newaxis, :]
    s.d2D_m = np.linalg.norm(xyz_cells[:, np.newaxis, :2] - xyz_UEs[np.newaxis, :, :2], axis=2)
    s.d3D_m = np.linalg.norm(xyz_cells[:, np.newaxis, :] - xyz_UEs[np.newaxis, :, :], axis=2)
    PL3D_RMa_LOS_PL1=(20 * np.log10(s.los1_t1 * s.d3D_m)) + (s.los1_t2 * np.log10(s.d3D_m)) + (s.los1_t3 + s.los1_t4 * s.d3D_m)
    a1=20 * np.log10(s.los1_t1) + s.los1_t3
    b1=20 * np.log10(s.los1_t1) + s.los1_t2
    c1=s.los1_t4
    if print_coeffs: print(f'RMa LOS PL1 Actual Coeffs:\ta={a1:g}, b={b1:g}, c={c1:g}')
    if return_PL1: 
       return PL3D_RMa_LOS_PL1
    PL3D_RMa_LOS_PL2=PL3D_RMa_LOS_PL1 + 40*np.log10(s.d3D_m/s.dBP)
    if return_PL2: 
       return PL3D_RMa_LOS_PL2
    PL3D_RMa_LOS = np.where(s.d2D_m < s.dBP, PL3D_RMa_LOS_PL1, PL3D_RMa_LOS_PL2)
    if s.LOS:
       if get_pathgain: 
          return 1.0 / np.power(10.0, PL3D_RMa_LOS/10.0) 
       if s.AIMM_simulator: 
          return PL3D_RMa_LOS[0]
       return PL3D_RMa_LOS
    else: 
       PL3D_RMa_NLOS = s.nlos_consts + s.nlos_t5 * (np.log10(s.d3D_m) - 3.0) 
       if get_pathgain: 
          return 1.0 / np.power(10.0, np.maximum(PL3D_RMa_LOS, PL3D_RMa_NLOS)/10.0)
       if s.AIMM_simulator: 
          return np.maximum(PL3D_RMa_LOS, PL3D_RMa_NLOS)[0]
       return np.maximum(PL3D_RMa_LOS, PL3D_RMa_NLOS)
# END RMa_pathloss

def test_RMa_pathloss_00():
  """
  Test function for the RMa_pathloss model.

  This function evaluates the RMa_pathloss model by calculating the path loss
  between a set of base station (BS) positions and user equipment (UE) positions
  in a non-line-of-sight (NLOS) scenario.

  The test uses:
  - Two BS positions with heights of 35.0 meters.
  - Three UE positions with heights of 1.5 meters.
  - A carrier frequency of 3.5 GHz.
  - A fixed BS height of 35.0 meters and UE height of 1.5 meters.

  The function prints the resulting path loss matrix and its shape.

  Returns:
    None
  """
  xyz_cells = np.array([[0.0, 0.0, 35.0],[8.0, 5.0, 35.0]])
  xyz_ues = np.array([[10.0, 10.0, 1.5], [2.0, 2.0, 1.5], [5.0, 5.0, 1.5]])
  PL = RMa_pathloss(fc_GHz=3.5, h_UT=1.5, h_BS=35.0, LOS=False)
  a=PL(xyz_cells, xyz_ues)
  print(f'PL3D_RMa_LOS=\n{a}; \tShape={a.shape}\n')

def test_RMa_pathloss_01():
  """
  Test function for the RMa_pathloss model.

  This function evaluates the RMa_pathloss model by calculating the path loss
  between a single base station (BS) and a range of user equipment (UE) positions
  distributed along the x-axis. The BS is fixed at a height of 35 meters, and the UEs
  are at a height of 1.5 meters. The path loss is computed for a non-line-of-sight (NLOS)
  scenario at a carrier frequency of 3.5 GHz.

  Steps:
  1. Define the position of the BS as a single point in 3D space.
  2. Generate 1000 UE positions along the x-axis using `np.linspace`, keeping the y-coordinate
     and height constant.
  3. Compute the path loss using the `RMa_pathloss` function.
  4. Print the resulting path loss values and their shape.

  Output:
    Prints the computed path loss values and their shape.

  Note:
    - The `RMa_pathloss` function is assumed to be defined elsewhere and should accept
      the parameters `fc_GHz`, `h_UT`, `h_BS`, and `LOS`, along with the positions of
      the BS and UEs.
    - The test is specifically designed for the RMa (Rural Macro) path loss model.
  """
  xyz_cells = np.array([[0.0, 0.0, 35.0]])  # Shape (M, 3) e.g. M=1
  # Create a range of UEs with linspace which vary only the x-coordinate
  x = np.linspace(10.0, 5000.0, 1000)  # 1000 UEs at different x-coordinates
  xyz_ues = np.column_stack((x, np.zeros_like(x), np.full_like(x, 1.5)))  # Shape (N, 3) e.g. N=1000
  PL = RMa_pathloss(fc_GHz=3.5, h_UT=1.5, h_BS=35.0, LOS=False)
  a=PL(xyz_cells, xyz_ues)
  print(f'PL3D_RMa_LOS=\n{a}; \tShape={a.shape}\n')
  # END test_RMa_pathloss_01

def plot(plot_type='pathloss', fc_GHz=3.5, h_UT=1.5, h_BS=35.0, zoom_box=False, print_10m_pl=False, author=' '):
    """
    Plot the 3GPP RMa pathloss or pathgain model predictions as a self-test.

    This function generates plots for the Rural Macro (RMa) pathloss or pathgain 
    models based on the 3GPP specifications. It supports both Line-of-Sight (LOS) 
    and Non-Line-of-Sight (NLOS) scenarios and includes options for zooming into 
    specific regions of the plot and printing pathloss values at 10 meters.

    Parameters:
      plot_type (str): Type of plot to generate. Options are:
               - 'pathloss': Plot pathloss in dB.
               - 'pathgain': Plot pathgain.
               Default is 'pathloss'.
      fc_GHz (float): Carrier frequency in GHz. Default is 3.5 GHz.
      h_UT (float): Height of the User Terminal (UT) in meters. Default is 1.5 m.
      h_BS (float): Height of the Base Station (BS) in meters. Default is 35.0 m.
      zoom_box (bool): Whether to include a zoomed-in inset of the plot. Default is False.
      print_10m_pl (bool): Whether to print the pathloss values at 10 meters for 
                 LOS, NLOS, and free-space conditions. Default is False.
      author (str): Author name to include in the plot timestamp. Default is an empty string.

    Returns:
      None: The function generates and saves the plot as PNG and PDF files.

    Notes:
      - The function uses the `RMa_pathloss` model to compute pathloss and pathgain values.
      - The free-space pathloss is also computed and plotted for comparison.
      - The zoomed-in inset is customizable and highlights a specific region of the plot.
      - The function saves the generated plots in the `../img/` directory with filenames 
        based on the plot type.

    Example:
      plot(plot_type='pathloss', fc_GHz=2.0, h_UT=1.5, h_BS=30.0, zoom_box=True, print_10m_pl=True, author='John Doe')
    """
    'Plot the pathloss model predictions, as a self-test.'
    import matplotlib.pyplot as plt
    from fig_timestamp_01 import fig_timestamp
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(color='gray', alpha=0.5, lw=0.5)
    xyz_cells = np.array([[0.0, 0.0, h_BS]])
    x = np.linspace(10.0, 5000.0, 4990)
    xyz_ues = np.column_stack((x, np.zeros_like(x), np.full_like(x, h_UT)))
    PL = RMa_pathloss(fc_GHz=fc_GHz, h_UT=h_UT, h_BS=h_BS, LOS=False)
    dBP_NLOS = PL.dBP
    dBP_NLOS_index = np.searchsorted(x, dBP_NLOS)
    PL_NLOS_dB = PL(xyz_cells, xyz_ues).squeeze()
    PG_NLOS = 1.0 / np.power(10.0, PL_NLOS_dB / 10.0)
    if plot_type == 'pathloss':
        ax.set_title(f'3GPP RMa pathloss models (dBP={dBP_NLOS:.0f}m)')
        ax.set_ylabel('pathloss (dB)')
        line=ax.plot(x, PL_NLOS_dB, lw=2, label=r'NLOS ($\sigma=8$)', color='blue')
        line_color = line[0].get_color()
        ax.vlines(dBP_NLOS, 0, PL_NLOS_dB[dBP_NLOS_index], line_color, 'dotted', lw=2)
        ax.fill_between(x, PL_NLOS_dB - 8, PL_NLOS_dB + 8, color=line_color, alpha=0.2)
        ax.set_ylim(50)
    else:
        ax.set_title(f'3GPP RMa pathgain models (dBP={dBP_NLOS:.0f}m)')
        ax.set_ylabel('pathgain')
        ax.plot(x, PG_NLOS, lw=2, label='NLOS pathgain')
    PL = RMa_pathloss(fc_GHz=fc_GHz, h_UT=h_UT, h_BS=h_BS, LOS=True)
    dBP_LOS = PL.dBP
    dBP_LOS_index = np.searchsorted(x, dBP_LOS)
    PL_LOS_dB = PL(xyz_cells, xyz_ues).squeeze()
    PG_LOS = 1.0 / np.power(10.0, PL_LOS_dB / 10.0)
    if plot_type == 'pathloss':
        line=ax.plot(x, PL_LOS_dB, lw=2, label=r'LOS ($\sigma=4$ before dBP, $\sigma=6$ after dBP)', color='orange')
        line_color = line[0].get_color()
        sigma = np.where(x < dBP_LOS, 4.0, 6.0)
        ax.vlines(dBP_LOS, 0, PL_LOS_dB[dBP_LOS_index], line_color, 'dotted', lw=2)
        ax.fill_between(x, PL_LOS_dB - sigma, PL_LOS_dB + sigma, color=line_color, alpha=0.2)
        ax.set_xlim(0, np.max(x))
        fnbase = '../img/RMa_pathloss_model'
    else:
        ax.plot(x, PG_LOS, lw=2, label='LOS pathgain')
        ax.set_ylim(0)
        ax.set_xlim(0, 1000)
        fnbase = '../img/RMa_pathgain_model'
    fs_pathloss_dB = (20 * np.log10(PL.d3D_m) + 20 * np.log10(fc_GHz*1e9) - 147.55).squeeze(axis=0)
    fs_pathloss = np.power(10.0, fs_pathloss_dB / 10.0)
    fs_pathgain = 1.0 / fs_pathloss
    if plot_type == 'pathloss':
        fspl_line = ax.plot(x, fs_pathloss_dB, lw=2, label='Free-space pathloss', color='red')
    else:
      fspl_line = ax.plot(x, fs_pathgain, lw=2, label='Free-space pathloss', color='red')
    if zoom_box:
        x1,x2,y1,y2 = 0.0, 50.1, 74, 78.0               # Define the area you want to zoom in on
        axins = ax.inset_axes([0.50, 0.1, 0.2, 0.33])   # Define where you want the zoom box to be placed
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
    ax.set_xlabel('distance (metres)')
    ax.legend()
    fig.tight_layout()
    if print_10m_pl:
        BLUE =   "\033[38;5;027m"
        ORANGE = "\033[38;5;202m"
        RED =    "\033[38;5;196m"
        RESET =  "\033[0m"
        print(f'\nPathloss at 10 metres:')
        print('----------------------')
        print(f'{BLUE}RMa-NLOS:       {PL_NLOS_dB[0]:.2f} dB')
        print(f'{ORANGE}RMa-LOS:        {PL_LOS_dB[0]:.2f} dB')
        print(f'{RED}Free-space:     {fs_pathloss_dB[0]:.2f} dB{RESET}\n')
    fig_timestamp(fig, rotation=0, fontsize=6, author=author)
    fig.savefig(f'{fnbase}.png')
    fig.savefig(f'{fnbase}.pdf')
    print(f'eog {fnbase}.png &')
    print(f'evince {fnbase}.pdf &')

if __name__ == '__main__':
  # plot(plot_type='pathgain') # simple self-test
  plot(plot_type='pathloss', zoom_box=True, print_10m_pl=True, author='Kishan Sthankiya') # simple self-test
