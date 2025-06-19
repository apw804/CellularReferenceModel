"""
CRM_energy_score_01.py
----------------------
This script provides tools for simulating a Cellular Reference Model (CRM) scenario and evaluating the energy efficiency of cellular systems. 

It is designed to methodically derive the energy score for each User Equipment (UE) in the system, defined as the number of bits transmitted per Joule expended.

Key Features
------------
- Scenario generation for cellular networks with configurable number of cells, UEs, and bandwidth.
- Calculation of energy efficiency (bits/Joule) for a given transmit power allocation.
- Utility functions to compute total throughput and power for all UEs or subsets (e.g., central cells).

Intended Use
------------
This script is intended for researchers and engineers analyzing the energy efficiency of cellular networks, especially in scenarios involving multiple cells and UEs. 

Author
------
Kishan Sthankiya, 2024-12-08

"""
import numpy as np

from CRM_standalone_05d import CRMStandalone
from dB_conversions_01 import dBm_to_watts
from cellular_reference_model_10d import CellularReferenceModel, CRMParameters

def get_scenario(seed=0, n_cells=19, n_ues=101, bw_MHz=10.0):
  """
  Generate a Cellular Reference Model (CRM) scenario with specified parameters.

  Parameters
  ----------
  seed : int, optional
    Random seed for reproducibility of UE locations. Default is 0.
  n_cells : int, optional
    Number of cells in the scenario. Default is 19.
  n_ues : int, optional
    Number of user equipments (UEs) in the scenario. Default is 101.
  bw_MHz : float, optional
    System bandwidth in MHz. Default is 10.0.

  Returns
  -------
  crm : CellularReferenceModel
    Configured CRM scenario with generated UE locations.

  Notes
  -----
  - The scenario uses a hexagonal cell layout and uniform UE distribution.
  - Pathloss model is set to 'UMa-NLOS' with a carrier frequency of 3.5 GHz.
  - Cell transmit power is set to 30 dBm.
  """
  params = CRMParameters(n_cells=n_cells,
                         n_ues=n_ues,
                         cell_layout='hex',
                         cell_radius_m=500,
                         UE_layout='uniform',
                         UE_radius=1500.00,
                         pathloss_model_name='UMa-NLOS',
                         radius_boost=0.0,
                         fc_GHz=3.5,
                         h_UT=1.5,
                         h_BS=25.0,
                         author='Kishan Sthankiya')
  params.bw_MHz=bw_MHz
  params.cell_tx_power_dBm=30.0
  crm = CellularReferenceModel(params, seed=seed)
  crm.generate_new_ue_locations(seed=seed)
  return crm

def energy_score(x0,crm_SA):
  """
  Calculates the energy efficiency score (bits per Joule) for a cellular system.

  The energy score is defined as the total number of bits transmitted to all connected UEs
  divided by the total transmit power used by the system. We use the radiated transmit 
  power (from dBm to watts) at the antenna as an analogue for the energy expended by the system. 
  
  The function handles the special case where no energy is expended by the central 7 cells, 
  returning a score of 0 to avoid division by zero or misleading results.

  Parameters
  ----------
  x0 : array-like
    Array of transmit powers for each cell in dBm.
  crm_SA : object
    System simulation object with methods:
      - set_tx_power_get_all_UE_throughputs(cell_tx_power_dBm): returns array of UE throughputs in Mbps.

  Returns
  -------
  float
    Energy efficiency score in bits per Joule.

  Notes
  -----
  - Only UEs with CQI > 0 are considered connected and included in throughput calculation.
  - The transmit power is converted from dBm to Watts using `dBm_to_watts`.
  - If all central 7 cells have zero transmit power, the function returns 0.0.
  """
  all_ues_throughputs_Mbps = crm_SA.set_tx_power_get_all_UE_throughputs(cell_tx_power_dBm=x0)
  cell_tx_watts_array = dBm_to_watts(x0)
  total_energy_expended = np.sum(cell_tx_watts_array)   
  if np.isclose(np.sum(cell_tx_watts_array[:7]), 0.0):
    total_energy_expended=0.0
    return 0.0
  energy_score = (1e6*np.sum(all_ues_throughputs_Mbps)) / total_energy_expended  
  return energy_score

def get_sum_throughput_and_power(x0, crm_SA):
  """
  Calculate the total throughput and total transmit power for all UEs in a CRM system simulation.

  Parameters
  ----------
  x0 : array-like or float
    Transmit power(s) in dBm for the cell(s).
  crm_SA : object
    CRM system simulation object with methods and attributes for throughput and cell information.

  Returns
  -------
  sum_all_ues_throughputs_Mbps : float
    The sum of throughputs for all UEs in Mbps.
  total_power_W : float
    The sum transmit power (Watts) for the cell(s) in the system.

  Notes
  -----
  Assumes the existence of a function `dBm_to_watts` that converts dBm to Watts.
  """
  all_ues_throughputs_Mbps = crm_SA.set_tx_power_get_all_UE_throughputs_vect(cell_tx_power_dBm=x0)
  sum_all_ues_throughputs_Mbps = np.sum(all_ues_throughputs_Mbps)
  cell_tx_watts_array = dBm_to_watts(x0)
  total_power_W = np.sum(cell_tx_watts_array)
  return sum_all_ues_throughputs_Mbps, total_power_W

def get_centre_cells_sum_throughput_and_system_power(x0, crm_SA):
  """
  Calculate the total throughput and system power for the central 7 cells in a CRM system simulation.

  Parameters
  ----------
  x0 : array-like or float
    Transmit power(s) in dBm for each cell.
  crm_SA : object
    CRM system simulation object with methods and attributes for throughput and cell information.

  Returns
  -------
  sum_centre_cells_throughputs_Mbps : float
    Sum of throughputs (in Mbps) for UEs attached to the central 7 cells.
  total_power_W : float
    Sum transmit power (in Watts) across all cells in the system.

  Notes
  -----
  - The function uses the `set_tx_power_get_all_UE_throughputs_vect` method of `crm_SA` to obtain UE throughputs.
  - Central cells are identified by `crm_SA.rsm['serving_cell'] <= 7`.
  - Power conversion from dBm to Watts is performed by `dBm_to_watts`.
  """
  all_ues_throughputs_Mbps = crm_SA.set_tx_power_get_all_UE_throughputs_vect(cell_tx_power_dBm=x0)
  centre_cells_throughputs_Mbps=np.where(crm_SA.rsm['serving_cell']<=7, all_ues_throughputs_Mbps, 0.0)
  sum_centre_cells_throughputs_Mbps=np.sum(centre_cells_throughputs_Mbps)
  cell_tx_watts_array = dBm_to_watts(x0)
  total_power_W = np.sum(cell_tx_watts_array)
  return sum_centre_cells_throughputs_Mbps, total_power_W

def get_all_ues_throughput_vect_and_system_power(x0, crm_SA):
  """
  Gets the throughput for all UEs as a vector and the total system power.

  Parameters
  ----------
  x0 : float or array-like
    The transmit power(s) in dBm for the cell(s).
  crm_SA : object
    CRM system simulation object with methods and attributes for throughput and cell information.

  Returns
  -------
  all_ues_throughputs_Mbps_vect : np.ndarray
    Throughput values (in Mbps) for all individual UEs.
  total_power_W : float
    Sum transmit power of the system in Watts.

  Notes
  -----
  Requires the function `dBm_to_watts` and NumPy as `np`.
  """
  all_ues_throughputs_Mbps_vect = crm_SA.set_tx_power_get_all_UE_throughputs_vect(cell_tx_power_dBm=x0)
  cell_tx_watts_array = dBm_to_watts(x0)
  total_power_W = np.sum(cell_tx_watts_array)
  return all_ues_throughputs_Mbps_vect, total_power_W


def get_CRM_SA(crm):
  """
  Create and initialize a CRMStandalone object with given CRM parameters and seed.

  Parameters
  ----------
  crm : object
    An object containing the parameters (`params`) and random seed (`seed`) required to initialize the CRMStandalone instance.

  Returns
  -------
  CRMStandalone
    An initialized CRMStandalone object with new user equipment (UE) locations generated.

  Notes
  -----
  The function assumes that the `crm` object has `params` and `seed` attributes,
  and that `CRMStandalone` and its `crm.generate_new_ue_locations` method are available.
  """
  crm_SA=CRMStandalone(params=crm.params, seed=crm.seed)
  crm_SA.crm.generate_new_ue_locations(seed=crm.seed)
  return crm_SA

def main(seed=0):# This is a self-test function
  """
  Runs a self-test for the CRM energy score calculation.

  Parameters
  ----------
  seed : int, optional
    Random seed for scenario generation (default is 0).

  Returns
  -------
  None
    This function prints the computed energy score and does not return a value.

  Notes
  -----
  This function generates a CRM scenario, computes the energy score using
  the default cell transmit power, and prints the result in bits/Joule.
  """
  crm = get_scenario(seed=seed)
  crm_SA = get_CRM_SA(crm)
  cell_tx_dBm_array=np.ones(crm_SA.crm.params.n_cells)*crm.params.cell_tx_power_dBm
  score=energy_score(cell_tx_dBm_array,crm_SA)
  print(f'energy score (bits/Joule):{score}')
  pass

  


if __name__ == '__main__':
  main()
  pass