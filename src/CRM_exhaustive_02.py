# Kishan Sthankiya 2025-01-21 
# - removed checkpoint file.
# - added individual UE throughputs.
# - testing for smaller output file format.
# Kishan Sthankiya 2024-12-17
# Latest working script with good performance.
# Script to run an exhaustive search to find the optimal energy score for a CRM simulation by varying the cell transmit power.
# python3 CRM_exhaustive_02.py

import sys
sys.path.append('../')
import time
import itertools
import numpy as np
import csv
from datetime import datetime as dt
from rich import print
from CRM_energy_score_01 import get_all_ues_throughput_vect_and_system_power, get_scenario, energy_score, get_CRM_SA

__version__='03'

def f_energy_score(x,crm_SA, fixed_power_cells_config_dBm=[-np.inf]*12):
  """
  Computes the energy score for a CRM configuration where the outer ring (12 cells)
  of a 19-cell hexagonal grid is set to -inf dBm by default, simulating these cells 
  being powered off or absent. The fixed_power_cells_config_dBm parameter allows 
  customization of these power values.

  Parameters
  ----------
  x : array-like
    Power values (in dBm) for the configurable (non-fixed) cells.
  crm_SA : object
    CRM Standalone system simulation object.
  fixed_power_cells_config_dBm : list of float, optional
    Power values (in dBm) for the 12 fixed outer cells. Defaults to all -inf.

  Returns
  -------
  float
    The calculated energy score for the complete configuration.

  Notes
  -----
  - Cells with indices 10, 14, and 18 are always set to a low power value (31 dBm).
  - The input `x` is concatenated with `fixed_power_cells_config_dBm` to form the 
    full power configuration.
  """
  low_power_dBm=31
  outer_cells_with_low_power=[10,14,18]
  fixed_power_cells=fixed_power_cells_config_dBm
  full_x=np.concatenate((x,fixed_power_cells))
  for cell_i in outer_cells_with_low_power:
     full_x[cell_i]=low_power_dBm
  return energy_score(full_x,crm_SA)

def f_alltp_sumtp_power_energyscore(x, crm_SA, fixed_power_cells_config_dBm=[-np.inf]*12, outer_cells_power_dBm=31, outer_cells=[10,14,18], individual_UE_throughputs=False):
  """
  Calculates the sum throughput, total power, and energy efficiency score for all UEs in the system.

  Parameters
  ----------
  x : array_like
    Power configuration vector for the cells to be optimized.
  crm_SA : object
    CRM Standalone system simulation object required for throughput and power calculation functions.
  fixed_power_cells_config_dBm : list of float, optional
    Fixed power values (in dBm) for non-optimized cells. Default is [-np.inf]*12.
  outer_cells_power_dBm : float, int, or list of float, optional
    Power value(s) (in dBm) to assign to the outer cells. If a single value is provided, it is applied to all outer cells.
  outer_cells : list of int, optional
    Indices of the outer cells to which `outer_cells_power_dBm` is applied. Default is [10, 14, 18] which are equally spaced in the outer ring.
  individual_UE_throughputs : bool, optional
    If True, returns individual UE throughputs in addition to aggregate metrics. Default is False.

  Returns
  -------
  sum_all_ues_throughputs_Mbps : float
    Total throughput (in Mbps) summed over all UEs in the system.
  power_W : float
    Total system power consumption (in Watts).
  energy_score_Mbps_per_Joule : float
    Energy efficiency score, defined as total throughput divided by total power (in Mbps/Joule).
  all_ues_tp_Mbps_vect : ndarray, optional
    Throughput (in Mbps) for each UE in the system. Returned only if `individual_UE_throughputs` is True.

  Notes
  -----
  - The function modifies the power configuration for specified outer cells.
  - Avoids divide-by-zero errors in energy score calculation by enforcing a minimum power value.
  """
  fixed_power_cells=fixed_power_cells_config_dBm
  full_x=np.concatenate((x,fixed_power_cells))

  if isinstance(outer_cells_power_dBm, (float,int)):
    for cell_i in outer_cells:
      full_x[cell_i]=outer_cells_power_dBm
  elif isinstance(outer_cells_power_dBm, list):
    for i, cell_i in enumerate(outer_cells):
      full_x[cell_i]=outer_cells_power_dBm[i]
  all_ues_tp_Mbps_vect, power_W = get_all_ues_throughput_vect_and_system_power(full_x,crm_SA) 
  sum_all_ues_throughputs_Mbps = np.sum(all_ues_tp_Mbps_vect)
  power_W=np.maximum(power_W, 1e-100) # Added to avoid divide-by-zero errors
  energy_score_Mbps_per_Joule = sum_all_ues_throughputs_Mbps/power_W
  if individual_UE_throughputs:
    return all_ues_tp_Mbps_vect, sum_all_ues_throughputs_Mbps, power_W, energy_score_Mbps_per_Joule
  return sum_all_ues_throughputs_Mbps, power_W, energy_score_Mbps_per_Joule

def exhaustive_search(seed=0, n_cells=7, n_ues=30, p_low_dBm=40, p_high_dBm=56, step_dBm=3, fixed_power_cells_config_dBm=[-np.inf]*12, outer_cells_power_dBm=31, outer_cells=[10,14,18], plot=False, individual_UE_throughputs=False):
  """
  Perform an exhaustive search over cell transmit power combinations to optimize throughput and energy efficiency.

  Parameters
  ----------
  seed : int, optional
    Random seed for scenario generation (default is 0).
  n_cells : int, optional
    Number of cells in the scenario (default is 7).
  n_ues : int, optional
    Number of user equipments (UEs) in the scenario (default is 30).
  p_low_dBm : int, optional
    Minimum transmit power in dBm for the search (default is 40).
  p_high_dBm : int, optional
    Maximum transmit power in dBm for the search (default is 56).
  step_dBm : int, optional
    Step size in dBm for transmit power levels (default is 3).
  fixed_power_cells_config_dBm : list of float, optional
    List specifying fixed transmit power (in dBm) for certain cells; use -np.inf for cells to be optimized (default is [-np.inf]*12).
  outer_cells_power_dBm : float, optional
    Transmit power (in dBm) for outer cells (default is 31).
  outer_cells : list of int, optional
    Indices of outer cells (default is [10, 14, 18]).
  plot : bool, optional
    If True, plot the scenario (default is False).
  individual_UE_throughputs : bool, optional
    If True, return individual UE throughputs in the results (default is False).

  Returns
  -------
  results : dict
    Dictionary mapping power combinations (tuple) to results:
    - If `individual_UE_throughputs` is False: (sum_throughput_Mbps, sum_power_W, energy_score_Mbps_Joule)
    - If `individual_UE_throughputs` is True: (all_ues_throughputs_Mbps_vect, sum_throughput_Mbps, sum_power_W, energy_score_Mbps_Joule)

  Notes
  -----
  This function may take a long time to run due to the exhaustive nature of the search.
  KeyboardInterrupt is handled to allow graceful interruption.
  """
  crm=get_scenario(seed=seed, n_cells=n_cells, n_ues=n_ues, bw_MHz=10.0)
  crm_SA=get_CRM_SA(crm)
  if plot:
    crm_SA.crm.plot(show_plot=False,show_voronoi=True,show_attachment=True,show_kilometres=True,show_pathloss_circles=True,fnbase=f'./exhaustive_search_seed{seed:04}')
  cells_tx_power=[0.0]*7
  power_levels=[-np.inf, *range(p_low_dBm, p_high_dBm+1, step_dBm)]
  power_combinations = list(itertools.product(power_levels, repeat=len(cells_tx_power)))
  results = {}
  for i, combination in enumerate(power_combinations):
    if combination in results:
      continue  # Skip already computed combinations
    try:
      x = np.array(combination)
      if individual_UE_throughputs:
        all_ues_throughputs_Mbps_vect, sum_throughput_Mbps, sum_power_W, energy_score_Mbps_Joule = f_alltp_sumtp_power_energyscore(x, crm_SA, fixed_power_cells_config_dBm, outer_cells_power_dBm, outer_cells, individual_UE_throughputs=True)
        results[combination] = (all_ues_throughputs_Mbps_vect, sum_throughput_Mbps, sum_power_W, energy_score_Mbps_Joule)
      else:
        sum_throughput_Mbps, sum_power_W, energy_score_Mbps_Joule = f_alltp_sumtp_power_energyscore(x, crm_SA,fixed_power_cells_config_dBm,outer_cells_power_dBm,outer_cells,individual_UE_throughputs=False)
        results[combination] = (sum_throughput_Mbps, sum_power_W, energy_score_Mbps_Joule)
    except KeyboardInterrupt:
      print("Interrupted! Aborting...")
  return results

def run_search(seed=0, n_cells=0, n_ues=0, p_low_dBm=0, p_high_dBm=0, step_dBm=0, fixed_power_cells_config_dBm=[-np.inf]*12, outer_cells_power_dBm=0, outer_cells=[10,14,18], out_dir='', individual_UE_throughputs=False):
  """
  Runs an exhaustive search to find the global optimal cell power configuration and saves all combination results to a CSV file.

  Parameters
  ----------
  seed : int, optional
    Random seed for reproducibility (default is 0).
  n_cells : int, optional
    Number of cells in the network (default is 0).
  n_ues : int, optional
    Number of user equipments (UEs) (default is 0).
  p_low_dBm : float, optional
    Lower bound for transmit power in dBm (default is 0).
  p_high_dBm : float, optional
    Upper bound for transmit power in dBm (default is 0).
  step_dBm : float, optional
    Step size for transmit power in dBm (default is 0).
  fixed_power_cells_config_dBm : list of float, optional
    List specifying fixed power configuration for each cell in dBm (default is [-np.inf]*12).
  outer_cells_power_dBm : float, optional
    Power setting for outer cells in dBm (default is 0).
  outer_cells : list of int, optional
    Indices of outer cells (default is [10, 14, 18]).
  out_dir : str, optional
    Output directory for saving the results CSV file (default is '').
  individual_UE_throughputs : bool, optional
    If True, saves individual UE throughputs in the results (default is False).

  Returns
  -------
  results_fn : str
    Filename of the saved CSV results file.

  Notes
  -----
  The function prints progress and timing information to the console. The results are saved in CSV format, with the filename including key parameters and the current date.
  """
  print(f'RUNNING: SEED{seed:04}')
  t0 = time.time()
  results = exhaustive_search(seed=seed, n_cells=n_cells, n_ues=n_ues, p_low_dBm=p_low_dBm, p_high_dBm=p_high_dBm, step_dBm=step_dBm, fixed_power_cells_config_dBm=fixed_power_cells_config_dBm, outer_cells_power_dBm=outer_cells_power_dBm, outer_cells=outer_cells, individual_UE_throughputs=individual_UE_throughputs)
  t1 = time.time()
  print(f"Total time taken: {t1 - t0} seconds")
  date_str=dt.now().strftime('%Y-%m-%d')
  results_fn=f'{date_str}-{n_cells}cells_{n_ues}ues_p{p_low_dBm}-{p_high_dBm}-ss{step_dBm:02}_seed{seed:04}.csv'
  results_path=f'./{out_dir}/{results_fn}'
  with open(results_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    if individual_UE_throughputs:
      writer.writerow(['Combination', 'Attached UEs Sum Throughput (Mbps)', 'Power Consumption (Watts)', 'Energy Score (Mbps/Joule)', 'Individual UEs Throughputs (Mbps)'])
      for combination, values in results.items():
        writer.writerow([str(combination), values[1], values[2], values[3], values[0]])
    else:
      writer.writerow(['Combination', 'Attached UEs Sum Throughput (Mbps)', 'Power Consumption (Watts)', 'Energy Score (Mbps/Joule)'])
      for combination, values in results.items():
        writer.writerow([str(combination), values[0], values[1], values[2]])
  print(f"Results saved to {results_fn}")
  print(f'COMPLETED: SEED {seed:04}')
  return results_fn

  
  

if __name__ == '__main__':
  run_search(seed=99, n_cells=19, n_ues=100,
              p_low_dBm=10, p_high_dBm=19,
              step_dBm=3,fixed_power_cells_config_dBm=[-np.inf]*12, 
              outer_cells_power_dBm=31, outer_cells=[10,14,18], out_dir='./')
  