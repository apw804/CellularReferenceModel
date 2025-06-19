# Kishan Sthankiya 2024-11-26 (v4b) change se_shannon_matrix
# Kishan Sthankiya 2024-11-19 (v4a) code cleanups.
# This is a standalone version of the CRM code that uses the vectorized operations to calculate the UE throughput.
# This is intended to run in place of the AIMM simulator.
# python3 CRM_standalone_04b.py

import sys
sys.path.append('../src')
import itertools
from time import time

import numpy as np

from color_text_00 import Color_text
from dB_conversions_01 import dBm_to_watts, watts_to_dBm, ratio_to_dB
from cellular_reference_model_10d import CellularReferenceModel, CRMParameters
from NR_5G_standard_functions_06d import _DMRS_RE, CQI_to_64QAM_efficiency, CQI_to_64QAM_efficiency_vect, SINR_to_CQI, get_nPRB_array,get_3GPP_data_rate_Mbps_vect_simple, MCS_TABLE_1_FOR_PDSCH_ARRAY

__version__ = '05d'
color_text=Color_text()
np.set_printoptions(precision=2, threshold=None, edgeitems=None, linewidth=200)

# PSC: Personal sanity check. A way to make sure the numbers look okay. Not neccessarily useful/used elsewhere.

class CRMStandalone:
  """
  Standalone Cellular Reference Model (CRM) optimization and evaluation class.

  This class provides methods for simulating, evaluating, and optimizing
  a cellular network using vectorized operations. It supports hill climbing
  and simulated annealing optimization for cell transmit power allocation.

  Parameters
  ----------
  params : CRMParameters
    CRM simulation parameters.
  seed : int, optional
    Random seed for reproducibility (default is 0).

  Attributes
  ----------
  params : CRMParameters
    CRM simulation parameters.
  crm : CellularReferenceModel
    Cellular reference model instance.
  bw_Hz : float
    System bandwidth in Hz.
  dbg : bool
    Debug flag.
  rng : np.random.Generator
    Random number generator.

  """
  def __init__(self, params, seed=0):
    self.params=params
    self.crm=CellularReferenceModel(params)
    self.bw_Hz=10e6 
    self.dbg=False
    self.rsm=None
    self.rng=np.random.default_rng(seed=seed)

  def set_formatting_preferences(self, ):
    """Set numpy print and error formatting preferences."""
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter}, edgeitems=30, linewidth=100000, precision=5)
    np.seterr(all='raise')

  def show_CRM_parameters(self, x):
    """Print CRM parameters for a given object."""
    if isinstance(x,CRMParameters):
      print(f'CRMParameters: {x.__dict__}')
    elif isinstance(x,CellularReferenceModel):
      print(f'CRMParameters: {x.params.__dict__}')
    else:
      raise ValueError('Invalid input type. Must be CRMParameters or CellularReferenceModel.')

  def get_distance_matrix(self, i_xyz, j_xyz):
    """
    Compute pairwise Euclidean distances between two sets of points.

    Parameters
    ----------
    i_xyz : np.ndarray
      Array of shape (N, 3) for N points.
    j_xyz : np.ndarray
      Array of shape (M, 3) for M points.

    Returns
    -------
    np.ndarray
      Distance matrix of shape (N, M).
    """
    return np.linalg.norm(i_xyz[:,np.newaxis,:]-j_xyz[np.newaxis,:,:],axis=2)

  def calculate_rsrp_threshold_dBm(self, thermal_noise_dBm_per_Hz=-173.85684411, bw_Hz=None):
    """
    Calculate RSRP threshold in dBm.

    Parameters
    ----------
    thermal_noise_dBm_per_Hz : float
      Thermal noise in dBm/Hz.
    bw_Hz : float
      Bandwidth in Hz.

    Returns
    -------
    float
      RSRP threshold in dBm.
    """
    return thermal_noise_dBm_per_Hz+10*np.log10(bw_Hz)

  def get_attachment_matrix_rsrp(self, rsrp_dBm_matrix, rsrp_threshold_dBm):
    """
    Compute UE attachment matrix based on RSRP threshold.

    Parameters
    ----------
    rsrp_dBm_matrix : np.ndarray
      RSRP values in dBm (cells x UEs).
    rsrp_threshold_dBm : float
      RSRP threshold in dBm.

    Returns
    -------
    np.ndarray
      Boolean attachment matrix (cells x UEs).
    """
    max_rsrp=np.max(rsrp_dBm_matrix,axis=0)
    max_rsrp_valid=np.greater(max_rsrp, rsrp_threshold_dBm)
    max_rsrp_idx=np.argmax(rsrp_dBm_matrix, axis=0)
    attachment_array_rsrp=np.where(max_rsrp_valid, max_rsrp_idx, -1.0)
    attachment_matrix_rsrp=attachment_array_rsrp==np.arange(rsrp_dBm_matrix.shape[0])[:,np.newaxis]
    return attachment_matrix_rsrp

  # def get_sinr_matrix(self, rsrp_W_matrix, sigma_W): # NEW
  #   """
  #   Compute SINR matrix given RSRP and noise.

  #   Parameters
  #   ----------
  #   rsrp_W_matrix : np.ndarray
  #     RSRP in Watts (cells x UEs).
  #   sigma_W : float
  #     Noise power in Watts.

  #   Returns
  #   -------
  #   np.ndarray
  #     SINR matrix (cells x UEs).
  #   """
  #   signal=rsrp_W_matrix
  #   total_power_per_UE=np.sum(rsrp_W_matrix, axis=0)
  #   interference=total_power_per_UE[np.newaxis,:]-signal
  #   denominator=interference+sigma_W
  #   denominator=np.maximum(denominator, 1e-100)
  #   return signal/denominator
  
 
  def get_sinr_matrix(self, rsrp_W_matrix, sigma_W): 
    """
    Compute SINR matrix given RSRP and noise.

    Parameters
    ----------
    rsrp_W_matrix : np.ndarray
      RSRP in Watts (cells x UEs).
    sigma_W : float
      Noise power in Watts.

    Returns
    -------
    np.ndarray
      SINR matrix (cells x UEs).
    """
    signal=rsrp_W_matrix
    column_sums=np.sum(rsrp_W_matrix, axis=0)
    interference=column_sums-signal # [2025-05-29] KS: This works because NumPy broadcasts column_sums across all rows automatically.
    noise=np.full_like(interference, sigma_W)
    denominator=interference+noise
    denominator=np.maximum(denominator, 1e-100)
    return signal/denominator

  def get_sir_matrix(self, rsrp_W_matrix):
    """
    Compute SIR matrix given RSRP.

    Parameters
    ----------
    rsrp_W_matrix : np.ndarray
      RSRP in Watts (cells x UEs).

    Returns
    -------
    np.ndarray
      SIR matrix (cells x UEs).
    """
    interference_W_matrix=rsrp_W_matrix.sum(axis=0)-rsrp_W_matrix
    return rsrp_W_matrix/interference_W_matrix

  def get_serving_cell_array(self, attachment_matrix):
    """
    Get serving cell index for each UE.

    Parameters
    ----------
    attachment_matrix : np.ndarray
      Boolean attachment matrix (cells x UEs).

    Returns
    -------
    np.ndarray
      Array of serving cell indices for each UE.
    """
    return np.where(
      np.any(attachment_matrix, axis=0),
      attachment_matrix.T.argmax(axis=1),
      -1
    )

  def get_unattached_ues(self, serving_cell_array):
    """
    Get indices of unattached UEs.

    Parameters
    ----------
    serving_cell_array : np.ndarray
      Array of serving cell indices for each UE.

    Returns
    -------
    np.ndarray
      Indices of unattached UEs.
    """
    return np.argwhere(serving_cell_array==-1).flatten()

  def get_n_unattached_ues(self, serving_cell_array):
    """
    Get number of unattached UEs.

    Parameters
    ----------
    serving_cell_array : np.ndarray
      Array of serving cell indices for each UE.

    Returns
    -------
    int
      Number of unattached UEs.
    """
    return np.count_nonzero(serving_cell_array==-1)

  def get_n_attached_ues_per_cell(self, attachment_matrix):
    """
    Get number of UEs attached to each cell.

    Parameters
    ----------
    attachment_matrix : np.ndarray
      Boolean attachment matrix (cells x UEs).

    Returns
    -------
    np.ndarray
      Number of attached UEs per cell.
    """
    return np.count_nonzero(attachment_matrix, axis=1)
  
  def get_radio_state_matrix_v2(self,
                  serving_cell_array,
                  serving_cell_sinr_dB_list,
                  cqi_list,
                  mcs_list):
    """
    Build radio state matrix for all UEs (vectorized).

    Parameters
    ----------
    serving_cell_array : np.ndarray
      Serving cell index for each UE.
    serving_cell_sinr_dB_list : array-like
      SINR in dB for each UE.
    cqi_list : array-like
      CQI index for each UE.
    mcs_list : array-like
      MCS index for each UE.

    Returns
    -------
    dict
      Radio state matrix fields for all UEs.
    """
    rsm={}
    rsm['serving_cell']=serving_cell_array.astype(int)
    rsm['serving_cell_bw_MHz'] = np.full(self.params.n_ues, self.bw_Hz / 1e6)
    rsm['serving_cell_sinr_dB']=np.array(serving_cell_sinr_dB_list)
    rsm['cqi']=np.array(cqi_list).astype(int)
    rsm['mcs']=np.array(mcs_list).astype(int)

    modulation_array = MCS_TABLE_1_FOR_PDSCH_ARRAY[mcs_list,1]
    rsm['modulation']=modulation_array

    code_rate_array = MCS_TABLE_1_FOR_PDSCH_ARRAY[mcs_list,2]
    rsm['code_rate']=code_rate_array

    rsm['v_layers']=np.ones(self.params.n_ues, dtype=np.int8)
    rsm['mu_numerology']=np.zeros(self.params.n_ues, dtype=np.int8)
    rsm['NRB_sc']=np.full(self.params.n_ues,12, dtype=np.int8)
    rsm['Nsh_symb']=14*np.power(2,rsm['mu_numerology'])
    rsm['NPRB_oh']=np.zeros(self.params.n_ues)
    rsm['nCC']=np.ones(self.params.n_ues, dtype=np.int8)
    rsm['NPRB_DMRS']=_DMRS_RE('type1','A',1,0)*np.ones(self.params.n_ues)

    nattached_ues_per_cell = np.bincount(serving_cell_array, minlength=self.params.n_cells)
    rsm['n_attached_ues_for_serving_cell'] = np.take(nattached_ues_per_cell, serving_cell_array).astype(int)

    rsm['serving_cell_max_nPRB']=get_nPRB_array(rsm['serving_cell_bw_MHz'], rsm['mu_numerology'])
    rsm['UE_nPRB_allocation']=np.floor_divide(rsm['serving_cell_max_nPRB'],rsm['n_attached_ues_for_serving_cell'])
    return rsm

  def init_model(self, params):
    """
    Initialize a new CellularReferenceModel.

    Parameters
    ----------
    params : CRMParameters
      CRM simulation parameters.

    Returns
    -------
    CellularReferenceModel
      New CRM model instance.
    """
    crm=CellularReferenceModel(params)
    return crm

  def compute_rsrp(self, cell_tx_power_W):
    """
    Compute RSRP in Watts and dBm for all UEs.

    Parameters
    ----------
    cell_tx_power_W : float or np.ndarray
      Cell transmit power(s) in Watts.

    Returns
    -------
    tuple of np.ndarray
      (RSRP in Watts, RSRP in dBm)
    """
    if isinstance(cell_tx_power_W, (int,float)):
      rsrp_W_matrix=self.crm.pathgain_matrix*cell_tx_power_W
    elif isinstance(cell_tx_power_W, np.ndarray):
      if cell_tx_power_W.ndim!=2:
        try:
          cell_tx_power_W=cell_tx_power_W.reshape(-1,1)
          rsrp_W_matrix=self.crm.pathgain_matrix*cell_tx_power_W
        except:
          raise ValueError('Invalid cell_tx_power_W array shape.')
      if cell_tx_power_W.shape[0]!=self.crm.params.n_cells:
        raise ValueError('Invalid cell_tx_power_W array shape.')
    else:
      raise ValueError('Invalid cell_tx_power_W type.')
    rsrp_dBm_matrix=watts_to_dBm(rsrp_W_matrix)
    return rsrp_W_matrix, rsrp_dBm_matrix

  def compute_sinr(self, rsrp_W_matrix):
    """
    Compute SINR and SINR in dB.

    Parameters
    ----------
    rsrp_W_matrix : np.ndarray
      RSRP in Watts (cells x UEs).

    Returns
    -------
    tuple of np.ndarray
      (SINR, SINR in dB)
    """
    sinr_matrix=self.get_sinr_matrix(rsrp_W_matrix, self.crm.sigma_W)
    sinr_dB_matrix=ratio_to_dB(sinr_matrix)
    return sinr_matrix, sinr_dB_matrix
  
  def set_tx_power_get_all_UE_throughputs(self, cell_tx_power_dBm=None):
    """
    Set cell transmit power and return all UE throughputs (loop version).

    Parameters
    ----------
    cell_tx_power_dBm : np.ndarray
      Transmit power for each cell in dBm.

    Returns
    -------
    np.ndarray
      Throughput for each UE (Mbps).
    """
    cell_tx_power_W=dBm_to_watts(cell_tx_power_dBm)
    rsrp_W_matrix, rsrp_dBm_matrix=self.compute_rsrp(cell_tx_power_W)
    rsrp_threshold_dBm=self.calculate_rsrp_threshold_dBm(bw_Hz=(self.params.bw_MHz*1e6))
    ue_serving_cell_array=np.argmax(rsrp_dBm_matrix, axis=0)
    sinr_matrix=self.get_sinr_matrix(rsrp_W_matrix, sigma_W=0.0)
    sinr_dB_matrix=ratio_to_dB(sinr_matrix)
    serving_cell_sinr_dB=[]
    cqi=[]
    mcs=[]
    se=[]
    for ue_i, cell_j in enumerate(ue_serving_cell_array):
      sc_sinr_dB_value = sinr_dB_matrix.T[ue_i][cell_j].item()
      if np.less(sc_sinr_dB_value, np.max(sinr_dB_matrix.T[ue_i])):
        print(f'Uh oh! Something went wrong with collecting UE[{ue_i}] SINR value.')
        print('Exiting...')
        exit()
      else:
        serving_cell_sinr_dB.append(sc_sinr_dB_value)
        cqi_i=SINR_to_CQI(sc_sinr_dB_value).item()
        mcs_i,se_i = CQI_to_64QAM_efficiency(cqi_i)
        cqi.append(cqi_i)
        mcs.append(mcs_i)
        se.append(se_i)
    self.rsm = self.get_radio_state_matrix_v2(ue_serving_cell_array, serving_cell_sinr_dB, cqi, mcs)
    throughputs_Mbps=get_3GPP_data_rate_Mbps_vect_simple(self.rsm)
    return throughputs_Mbps
  
  def set_tx_power_get_all_UE_throughputs_vect(self, cell_tx_power_dBm=None):
    """
    Set cell transmit power and return all UE throughputs (vectorized).

    Parameters
    ----------
    cell_tx_power_dBm : np.ndarray
      Transmit power for each cell in dBm.

    Returns
    -------
    np.ndarray
      Throughput for each UE (Mbps).
    """
    cell_tx_power_W=dBm_to_watts(cell_tx_power_dBm)
    rsrp_W_matrix, rsrp_dBm_matrix=self.compute_rsrp(cell_tx_power_W)
    ue_serving_cell_array=np.argmax(rsrp_dBm_matrix, axis=0)
    sinr_matrix=self.get_sinr_matrix(rsrp_W_matrix, sigma_W=0.0)
    sinr_dB_matrix=ratio_to_dB(sinr_matrix)
    serving_cell_indices=(ue_serving_cell_array,np.arange(sinr_dB_matrix.shape[1]))
    serving_cell_sinr_dB=sinr_dB_matrix[serving_cell_indices]
    if not np.all(serving_cell_sinr_dB >= np.max(sinr_dB_matrix, axis=0)):
      raise ValueError('Unexpected SINR value detected for one or more UEs.')
    cqi=SINR_to_CQI(serving_cell_sinr_dB)
    mcs, se = CQI_to_64QAM_efficiency_vect(cqi)
    self.rsm = self.get_radio_state_matrix_v2(ue_serving_cell_array, serving_cell_sinr_dB, cqi, mcs)
    throughputs_Mbps=get_3GPP_data_rate_Mbps_vect_simple(self.rsm)
    return throughputs_Mbps


def test_brute_force_search():
  t0=time()
  params=CRMParameters(n_cells=19, n_ues=100, cell_radius_m=500, UE_layout='uniform', radius_boost=0.0, pathloss_model_name='UMa-NLOS', h_UT=2.0, h_BS=25.0, author='Kishan Sthankiya')
  crm_SA=CRMStandalone(params)
  p_low_dBm = 3
  p_high_dBm = 10
  step_dBm = 3
  cells_tx_power=[0.0]*7
  power_levels=[-np.inf, *range(p_low_dBm, p_high_dBm+1, step_dBm)]
  power_combinations = list(itertools.product(power_levels, repeat=len(cells_tx_power)))
  for i, power_combination in enumerate(power_combinations):
    cells_tx_power[:7] = power_combination
    x=np.concatenate((cells_tx_power, [0]*12))
    print(f'{i+1}: {np.sum(crm_SA.set_tx_power_get_all_UE_throughputs_vect(x))}')
  t1=time()
  print(f'Time elapsed: {t1-t0:.8f} seconds')

def test_CRM_standalone_lite():
  t0=time()
  params=CRMParameters(n_cells=3, n_ues=4, cell_radius_m=500, UE_layout='uniform', radius_boost=0.0, pathloss_model_name='UMa-NLOS', h_UT=2.0, h_BS=25.0, author='Kishan Sthankiya')
  crm_SA=CRMStandalone(params)
  cells_tx_power_dBm=np.array([3.0,9.0,5.0])
  print(f'System throughput (Mbps): {np.sum(crm_SA.set_tx_power_get_all_UE_throughputs_vect(cells_tx_power_dBm))}')
  t1=time()
  print(f'Time elapsed: {t1-t0:.8f} seconds')
  

if __name__=='__main__':
  test_CRM_standalone_lite()