"""
Cellular Reference Model (CRM) optimization using metaheuristic algorithms.

This module implements random search, hill climbing, and simulated annealing
algorithms for optimizing transmit power allocation in cellular networks to
maximize energy efficiency (throughput per unit power consumption).

Functions
---------
f_tp_power_energy_score : 
    Calculate throughput, power consumption, and energy efficiency.
generate_neighbours :
    Generate neighboring power configurations for local search.
vanilla_hillclimb :
    Basic hill climbing optimization algorithm.
random_restart_hillclimb :
    Hill climbing with multiple random restarts.
simulated_annealing :
    Simulated annealing optimization with configurable cooling schedules.
get_random_cell_tx_power :
    Generate random neighbor by modifying one cell's power.
standard_cooling, linear_cooling, exponential_cooling, logarithmic_cooling :
    Temperature cooling schedules for simulated annealing.
run_random_restart_hillclimb :
    Execute hill climbing with CSV export and logging.
run_simulated_annealing :
    Execute simulated annealing with CSV export and logging.

Notes
-----
The optimization algorithms work with power levels in dBm, including -np.inf 
to represent disabled (i.e. powered OFF) cells. Results are exported to timestamped 
CSV files with comprehensive optimization statistics.

Author: Kishan Sthankiya
Date: 2025-04-25
Version: 03
"""


import sys
sys.path.append('../')
import time
import logging
import numpy as np
import csv
from datetime import datetime as dt
from rich import print
from CRM_energy_score_01 import get_all_ues_throughput_vect_and_system_power, get_scenario, get_CRM_SA

__version__='03'

# Set up logging to stream to a file
logging.basicConfig(
    filename="CRM_random_03.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def f_tp_power_energy_score(x,crm_SA, fixed_power_cells_config_dBm=[-np.inf]*12,outer_cells_power_dBm=31,outer_cells=[10,14,18]):
  """
    Calculate throughput, power consumption, and energy efficiency score for a CRM system.
    
    This function computes the total system throughput, power consumption, and energy 
    efficiency (throughput per unit power) for all UEs in a cellular network system
    with configurable cell power settings.
    
    Parameters
    ----------
    x : array_like
      Power configuration vector for variable power cells (in dBm).
    crm_SA : object
      CRM system analyzer object containing network configuration and methods
      for throughput and power calculations.
    fixed_power_cells_config_dBm : list of float, optional
      Power configuration for fixed power cells in dBm. Default is [-np.inf]*12,
      representing 12 cells with minimum power.
    outer_cells_power_dBm : float or list of float, optional
      Power setting(s) for outer cells in dBm. If float, same power is applied
      to all outer cells. If list, individual power for each outer cell.
      Default is 31 dBm.
    outer_cells : list of int, optional
      Indices of outer cells in the system. Default is [10, 14, 18].
    
    Returns
    -------
    sum_all_ues_throughputs_Mbps : float
      Total throughput of all UEs in the system (Mbps).
    power_W : float
      Total system power consumption (Watts). Minimum value is 1e-100 to
      prevent division by zero.
    energy_score_Mbps_per_Joule : float
      Energy efficiency score calculated as total throughput divided by
      total power consumption (Mbps/Joule).
    
    Notes
    -----
    The function constructs a complete power configuration by concatenating
    the variable power vector `x` with fixed power settings, then applies
    outer cell power configurations. The energy score represents the system's
    energy efficiency.
    
    Examples
    --------
    >>> x = [20, 25, 30]  # Power for variable cells
    >>> throughput, power, efficiency = f_tp_power_energy_score(x, crm_analyzer)
    >>> print(f"Efficiency: {efficiency:.2f} Mbps/Joule")
    """
  fixed_power_cells=fixed_power_cells_config_dBm
  full_x=np.concatenate((x,fixed_power_cells))
  if isinstance(outer_cells_power_dBm, (float,int)):
    for cell_i in outer_cells:
      full_x[cell_i]=outer_cells_power_dBm
  elif isinstance(outer_cells_power_dBm, list):
    for i, cell_i in enumerate(outer_cells):
      full_x[cell_i]=outer_cells_power_dBm[i]   
  all_ues_throughputs_Mbps_vect, total_power_W = get_all_ues_throughput_vect_and_system_power(full_x,crm_SA) # For all ues in the system
  power_W=np.maximum(total_power_W, 1e-100) # Added to avoid divide-by-zero errors
  sum_all_ues_throughputs_Mbps= np.sum(all_ues_throughputs_Mbps_vect)
  energy_score_Mbps_per_Joule = sum_all_ues_throughputs_Mbps/power_W 
  return sum_all_ues_throughputs_Mbps, power_W, energy_score_Mbps_per_Joule


def generate_neighbours(array, valid_values, step):
    """
    Generate neighboring configurations by modifying power levels in an array.
    
    This function creates all possible neighboring states by incrementing or 
    decrementing power values by a specified step size, while handling special 
    cases for minimum and maximum power levels.
    
    Parameters
    ----------
    array : numpy.ndarray
      Input array containing current power levels. May contain -np.inf 
      values representing disabled states.
    valid_values : array_like
      List or array of valid power values in ascending order. Expected 
      to include numerical power levels (e.g., [10, 13, 16, 19] dBm).
    step : float or int
      Step size for incrementing/decrementing power levels.
    
    Returns
    -------
    list of numpy.ndarray
      List of neighboring arrays, where each neighbor differs from the 
      input array by exactly one modified element.
    
    Notes
    -----
    Special handling is implemented for edge cases:
    - When all elements are -np.inf: neighbors set first valid power level
    - When all elements are at maximum (19.0): neighbors step down from maximum
    - For individual elements: handles transitions to/from -np.inf state
    
    The function ensures that:
    - Generated neighbors contain only valid power values or -np.inf
    - No duplicate of the original array is included in neighbors
    - Boundary conditions are properly managed
    
    Examples
    --------
    >>> import numpy as np
    >>> array = np.array([10, 13, -np.inf])
    >>> valid_values = [10, 13, 16, 19]
    >>> step = 3
    >>> neighbors = generate_neighbours(array, valid_values, step)
    """
    neighbours = []
    if np.all(array == -np.inf):
      for i,num in enumerate(array):
        potential_neighbour=array.copy()
        potential_neighbour[i]=valid_values[1] # Sets the value to 10 dBm
        neighbours.append(potential_neighbour)
      return neighbours
    if np.all(array == 19.0):
      for i,num in enumerate(array):
        potential_neighbour=array.copy()
        potential_neighbour[i]=valid_values[-2] # Sets the value to 16 dBm
        neighbours.append(potential_neighbour)
      return neighbours
    else:
      for i,num in enumerate(array):
          if num+step in valid_values:
            potential_neighbour=array.copy()
            potential_neighbour[i]=num+step
            if not np.all(potential_neighbour==array):
              neighbours.append(potential_neighbour)
          if num-step in valid_values:
            potential_neighbour=array.copy()
            potential_neighbour[i]=num-step
            if not np.all(potential_neighbour==array):
              neighbours.append(potential_neighbour)
          if num-step < valid_values[1]:
          # if num-step < valid_values[0]:
            potential_neighbour=array.copy()
            potential_neighbour[i]=-np.inf
            # For the -inf case, if it is already -inf, we don't want to add this combination to the neighbours.
            if not np.all(potential_neighbour==array):
              neighbours.append(potential_neighbour)
          if num+step < valid_values[1]: # This is the case if `num` is -inf. This allows us to step back to 10 dBm if num is -inf.
          # if num+step < valid_values[0]: # This is the case if `num` is -inf
            potential_neighbour=array.copy()
            potential_neighbour[i]=valid_values[1]
            if not np.all(potential_neighbour==array):
              neighbours.append(potential_neighbour)
    return neighbours

def vanilla_hillclimb(seed=0, n_cells=7, n_ues=30, p_low_dBm=40.0, p_high_dBm=56.0, step_dBm=3.0,fixed_power_cells_config_dBm=[-np.inf]*12, outer_cells_power_dBm=31, outer_cells=[10,14,18]):
  """
  Perform vanilla hill climbing optimization for cellular power allocation.
  This function implements a hill climbing algorithm to optimize transmit power
  levels for cellular base stations in a CRM (Cellular Reference Model) scenario.
  The algorithm starts with a random initial power configuration and iteratively
  moves to neighboring configurations that improve the objective function until
  a local optimum is reached.

  Parameters
  ----------
  seed : int, optional
    Random seed for reproducible results, by default 0
  n_cells : int, optional
    Number of cells in the scenario, by default 7
  n_ues : int, optional
    Number of user equipments (UEs) in the scenario, by default 30
  p_low_dBm : float, optional
    Lower bound for transmit power levels in dBm, by default 40.0
  p_high_dBm : float, optional
    Upper bound for transmit power levels in dBm, by default 56.0
  step_dBm : float, optional
    Step size for power level increments in dBm, by default 3.0
  fixed_power_cells_config_dBm : list, optional
    Fixed power configuration for cells in dBm. List of 12 values where
    -np.inf indicates variable power cells, by default [-np.inf]*12
  outer_cells_power_dBm : float, optional
    Power level for outer cells in dBm, by default 31
  outer_cells : list, optional
    Indices of outer cells, by default [10,14,18]

  Returns
  -------
  dict
    Dictionary containing optimization results for each iteration round.
    Keys are iteration numbers (int), values are dictionaries mapping
    power configuration strings to numpy arrays of objective function values
    [throughput, power, energy_score].

  Notes
  -----
  The algorithm uses a hill climbing approach which:
  1. Starts with a random initial power configuration
  2. Generates all possible neighboring configurations by varying one cell's
     power by ±step_dBm
  3. Evaluates the objective function for all neighbors
  4. Moves to the neighbor with the largest positive improvement
  5. Terminates when no neighbor provides improvement (local optimum)
  The objective function combines throughput, power consumption, and energy
  efficiency metrics. A checkpoint file is created to save intermediate results.

  Examples
  --------
  >>> results = vanilla_hillclimb(seed=42, n_cells=7, n_ues=20)
  >>> print(f"Optimization completed in {len(results)} iterations")
  See Also
  --------
  get_scenario : Creates the CRM scenario
  get_CRM_SA : Gets the CRM simulated annealing object
  f_tp_power_energy_score : Objective function for power optimization
  generate_neighbours : Generates neighboring power configurations
  """
  
  checkpoint_file=f"random_vanilla_hillclimb_checkpoint_{n_cells}cells_{n_ues}ues_{p_low_dBm}_{p_high_dBm}_{step_dBm}_seed{seed:03}.csv"

  crm=get_scenario(seed=seed, n_cells=n_cells, n_ues=n_ues, bw_MHz=10.0)
  crm_SA=get_CRM_SA(crm)
  results={}

  power_levels=[-np.inf, *np.arange(p_low_dBm, p_high_dBm+1, step_dBm)]
  initial_cell_tx_powers=crm_SA.rng.choice(power_levels, size=7)  # Size is 7 for the central 7 cells.

  f_x_opt=f_tp_power_energy_score(x=initial_cell_tx_powers,crm_SA=crm_SA,fixed_power_cells_config_dBm=fixed_power_cells_config_dBm,outer_cells_power_dBm=outer_cells_power_dBm,outer_cells=outer_cells)
  
  iteration_round=0
  
  # Store the initial combination to the results dictionary
  results[iteration_round]={np.array2string(initial_cell_tx_powers): np.array(f_x_opt)}
  x_opt=initial_cell_tx_powers

  while True:
    x_opt_neighbours=generate_neighbours(x_opt, valid_values=power_levels, step=step_dBm)
    f_x_neighbours=np.zeros((len(x_opt_neighbours),3))
    for i, neighbour in enumerate(x_opt_neighbours):
      f_x_neighbours[i]=f_tp_power_energy_score(x=neighbour,crm_SA=crm_SA,fixed_power_cells_config_dBm=fixed_power_cells_config_dBm,outer_cells_power_dBm=outer_cells_power_dBm,outer_cells=outer_cells)
      # Ensure results dictionary is initialized for the current iteration_round
      if iteration_round not in results:
        results[iteration_round]={}
      
      # Update the results dictionary with the neighbour and its corresponding value
      results[iteration_round][np.array2string(neighbour)] = f_x_neighbours[i]

    # Calculate the deltas
    delta_f_x_neighbours=f_x_neighbours[:,-1]-f_x_opt[-1]

    # What is the index of the neighbour that has the largest positive delta?
    delta_max_idx=np.argmax(delta_f_x_neighbours)

    # What is the array for the delta_max neighbour?
    delta_max_neighbour_array=f_x_neighbours[delta_max_idx]

    # If the largest delta_max is a negative value, then we have reached an optimum.
    if np.less(delta_f_x_neighbours[delta_max_idx],0.0): #  CAREFUL! Might need less_equal to avoid infinite loop for -inf cases.
      print(f'Optimum found.')
      return results
    
    else:
      f_x_opt=delta_max_neighbour_array.copy()
      x_opt=x_opt_neighbours[delta_max_idx].copy()
      iteration_round+=1

def random_restart_hillclimb(n_evals=1,max_iterations=None,seed=0, n_cells=7, n_ues=30, p_low_dBm=40, p_high_dBm=56, step_dBm=3,fixed_power_cells_config_dBm=[-np.inf]*12,outer_cells_power_dBm=31,outer_cells=[10,14,18],):
  """
    Perform random restart hill climbing optimization for cellular power allocation.
    This function implements a random restart hill climbing algorithm to optimize
    power allocation in a cellular network scenario. It performs multiple restarts
    with random initial configurations and uses hill climbing to find local optima.

    Parameters
    ----------
    n_evals : int, optional
      Number of evaluation rounds (restarts) to perform, by default 1
    max_iterations : int, optional
      Maximum number of global iterations across all restarts, by default None
    seed : int, optional
      Random seed for reproducibility, by default 0
    n_cells : int, optional
      Number of central cells in the scenario, by default 7
    n_ues : int, optional
      Number of user equipments in the scenario, by default 30
    p_low_dBm : int, optional
      Lower bound of power levels in dBm, by default 40
    p_high_dBm : int, optional
      Upper bound of power levels in dBm, by default 56
    step_dBm : int, optional
      Step size for power level increments in dBm, by default 3
    fixed_power_cells_config_dBm : list, optional
      Fixed power configuration for cells in dBm, by default [-np.inf]*12
    outer_cells_power_dBm : int, optional
      Power level for outer cells in dBm, by default 31
    outer_cells : list, optional
      Indices of outer cells, by default [10,14,18]
      
    Returns
    -------
    dict
      Nested dictionary containing optimization results with structure:
      {eval_round: {iteration: {power_config_string: score_array}}}
      where:
      - eval_round: restart round number
      - iteration: iteration number within the restart
      - power_config_string: string representation of power configuration
      - score_array: numpy array with [throughput, power, energy_score]

    Notes
    -----
    The algorithm performs the following steps:
    1. Initialize random power configuration for central cells
    2. Evaluate initial configuration using f_tp_power_energy_score
    3. Generate neighbors by varying power levels
    4. Select neighbor with maximum positive improvement
    5. Repeat until local optimum is reached or max_iterations exceeded
    6. Restart with new random configuration for remaining evaluations
    The optimization stops when either all n_evals are completed or 
    max_iterations is reached globally across all restarts.

    Examples
    --------
    >>> results = random_restart_hillclimb(n_evals=3, max_iterations=100, seed=42)
    >>> # Access results for first restart, second iteration
    >>> first_restart_results = results[0]
    """
  
  crm=get_scenario(seed=seed, n_cells=n_cells, n_ues=n_ues, bw_MHz=10.0)
  crm_SA=get_CRM_SA(crm)

  random_restart_results={}
  eval_round=0
  global_iterations=0
  power_levels=[-np.inf, *np.arange(p_low_dBm, p_high_dBm+1, step_dBm)]

  while n_evals>0:
    random_restart_results[eval_round]={}
    initial_cell_tx_powers=crm_SA.rng.choice(power_levels, size=7)  # Size is 7 for the central 7 cells.

    f_x_opt=f_tp_power_energy_score(x=initial_cell_tx_powers,crm_SA=crm_SA,fixed_power_cells_config_dBm=fixed_power_cells_config_dBm,outer_cells_power_dBm=outer_cells_power_dBm,outer_cells=outer_cells)
    
    iteration=0
    random_restart_results[eval_round]={iteration:{}}
    random_restart_results[eval_round][iteration]={np.array2string(initial_cell_tx_powers): np.array(f_x_opt)}

    x_opt=initial_cell_tx_powers

    if global_iterations>=max_iterations: break
    
    while global_iterations<max_iterations:
      iteration+=1
      global_iterations+=1
      if global_iterations>=max_iterations: break

      x_opt_neighbours=generate_neighbours(x_opt, valid_values=power_levels, step=step_dBm)

      f_x_neighbours=np.zeros((len(x_opt_neighbours),3))
      for i, neighbour in enumerate(x_opt_neighbours):
        f_x_neighbours[i]=f_tp_power_energy_score(x=neighbour,
                                      crm_SA=crm_SA,
                                      fixed_power_cells_config_dBm=fixed_power_cells_config_dBm,
                                      outer_cells_power_dBm=outer_cells_power_dBm,
                                      outer_cells=outer_cells)

      # Calculate the deltas
      delta_f_x_neighbours=f_x_neighbours[:,-1]-f_x_opt[-1]

      # If the largest delta_max is a negative value, then we have reached an optimum.
      if np.all(delta_f_x_neighbours<0.0):
        break

      # What is the index of the neighbour that has the largest positive delta?
      delta_max_idx=np.argmax(delta_f_x_neighbours)

      # What is the array for the delta_max neighbour?
      delta_max_neighbour_array=f_x_neighbours[delta_max_idx]

      # What is the combination for the delta_max neighbour?
      delta_max_neighbour_combination=x_opt_neighbours[delta_max_idx]

      # Now we log this to the results
      random_restart_results[eval_round][iteration]={np.array2string(delta_max_neighbour_combination): np.array(delta_max_neighbour_array)}
      
      # Update what the x_opt and f_x_opt should be (with a copy)
      f_x_opt=delta_max_neighbour_array.copy()
      x_opt=delta_max_neighbour_combination.copy()
      
    eval_round+=1
    n_evals-=1
    if global_iterations<max_iterations: n_evals+=1 # Ignores the n_evals and continues.
    if global_iterations>=max_iterations: break
      
  return random_restart_results

def get_random_cell_tx_power(crm_SA, valid_power_levels, cell_tx_array, x_opt):
  """
  Generate a random neighbor solution by modifying the transmit power of a randomly selected cell.
  
  This function implements a neighborhood generation strategy for optimization algorithms
  by randomly selecting a cell from valid power levels and assigning it a random
  transmit power value from the available power array.
  
  Parameters
  ----------
  crm_SA : object
    CRM Standalone object containing a random number generator (rng attribute).
  valid_power_levels : array_like
    Array of valid cell indices that can have their power levels modified.
  cell_tx_array : array_like
    Array of available transmit power levels. The length determines the range
    of possible power indices (0 to len(cell_tx_array)-1).
  x_opt : array_like
    Current solution vector representing transmit power assignments for all cells.
    Will be copied to create the neighbor solution.
  
  Returns
  -------
  ndarray
    A copy of x_opt with one randomly selected cell's transmit power modified
    to a random value from the valid range.
  
  Notes
  -----
  The function modifies only one cell's power level per call, making it suitable
  for local search and simulated annealing algorithms that require small
  perturbations to generate neighbor solutions.
  """
  rand_cell=crm_SA.rng.choice(a=valid_power_levels)
  rand_tx_power = crm_SA.rng.choice(a=len(cell_tx_array))
  x_neighbour=x_opt.copy()
  x_neighbour[rand_cell]=rand_tx_power
  return x_neighbour


def standard_cooling(temperature, alpha):
  """
  Apply standard cooling schedule for simulated annealing.

  This function implements a geometric cooling schedule where the temperature
  is reduced by multiplying with a cooling factor (alpha).

  Parameters
  ----------
  temperature : float
    Current temperature value. Must be positive.
  alpha : float
    Cooling factor used to reduce temperature. Should be between 0 and 1
    for proper cooling (typically 0.8-0.99).

  Returns
  -------
  float
    New temperature after applying cooling schedule.

  Notes
  -----
  The standard cooling schedule follows the formula:
    T_new = T_current * alpha
  """
  return temperature*alpha

def linear_cooling(T_start, T_end, max_n_iterationss, iteration):
  """
  Calculate temperature for linear cooling schedule in simulated annealing.

  This function implements a linear cooling schedule where the temperature 
  decreases linearly from a starting temperature to an ending temperature 
  over a specified number of iterations.

  Parameters
  ----------
  T_start : float
    The initial temperature at the beginning of the cooling schedule.
  T_end : float
    The final temperature at the end of the cooling schedule.
  max_n_iterationss : int
    The maximum number of iterations over which cooling occurs.
    Note: Parameter name appears to have a typo ('iterationss').
  iteration : int
    The current iteration number (0-based indexing expected).

  Returns
  -------
  float
    The calculated temperature for the given iteration.

  Notes
  -----
  The cooling rate (alpha) is calculated as (T_start - T_end) / max_n_iterationss.
  The temperature at any iteration is: T_start - (alpha * iteration).
  """
  alpha=(T_start-T_end)/max_n_iterationss
  return T_start - (alpha * iteration)

def exponential_cooling(T_start, alpha, iteration):
    """
    Exponential cooling function for simulated annealing algorithms.

    Calculates the temperature at a given iteration using exponential decay,
    with a lower bound to prevent numerical underflow.

    Parameters
    ----------
    T_start : float
      Initial temperature value. Must be positive.
    alpha : float
      Cooling rate parameter. Should be between 0 and 1 for proper cooling.
      Values closer to 1 result in slower cooling.
    iteration : int
      Current iteration number. Must be non-negative.

    Returns
    -------
    float
      Current temperature value, bounded below by 1e-20 to prevent
      numerical underflow.

    Notes
    -----
    The temperature follows the formula: T = max(T_start * alpha^iteration, 1e-20.
    """
    return max(T_start * (alpha ** iteration), 1e-20)

def logarithmic_cooling(T_start, T_end, max_n_iterations, iteration):
  """
  Calculate temperature for logarithmic cooling schedule in simulated annealing.
  
  This function implements a logarithmic cooling schedule that decreases temperature
  from an initial value to a final value over a specified number of iterations.
  The cooling follows a logarithmic decay pattern, providing slower cooling
  compared to exponential schedules.
  
  Parameters
  ----------
  T_start : float
    Initial temperature at the beginning of the cooling process.
  T_end : float
    Final temperature at the end of the cooling process.
  max_n_iterations : int
    Maximum number of iterations for the cooling schedule.
  iteration : int
    Current iteration number (0-based indexing expected).
    
  Returns
  -------
  float
    Current temperature value for the given iteration.
    
  Notes
  -----
  The cooling schedule follows the formula:
  T(i) = T_end + (T_start - T_end) / (1 + k * log(1 + i))
  where k = (T_start - T_end) / log(1 + max_n_iterations)
  """
  k=(T_start-T_end) / np.log(1+max_n_iterations)
  return T_end + (T_start-T_end) / (1 + k*np.log(1+iteration))


def simulated_annealing(max_n_iterations=100,
                        T_start=1000,
                        T_end=0.1,
                        cooling_schedule='linear',
                        alpha=0.1,
                        seed=0, 
                        n_cells=7, 
                        n_ues=30, 
                        p_low_dBm=40, 
                        p_high_dBm=56, 
                        step_dBm=3,
                        fixed_power_cells_config_dBm=[-np.inf]*12,
                        outer_cells_power_dBm=31,
                        outer_cells=[10,14,18],
                        verbose=False):
  """
  Optimize cellular network power allocation using simulated annealing algorithm.
  
  This function implements a simulated annealing optimization algorithm to find
  optimal transmit power levels for cellular base stations in a CRM (Cellular Reference Model) scenario. The algorithm uses probabilistic acceptance
  of worse solutions to escape local optima, with acceptance probability
  decreasing as temperature cools according to the specified cooling schedule.
  
  Parameters
  ----------
  max_n_iterations : int, optional
      Maximum number of iterations to run the algorithm, by default 100.
      Each iteration generates and evaluates one candidate solution.
  T_start : float, optional
      Initial temperature in Kelvin for the annealing process, by default 1000.
      Higher values increase initial acceptance of worse solutions.
  T_end : float, optional
      Final temperature in Kelvin, by default 0.1. Used only for linear
      and logarithmic cooling schedules.
  cooling_schedule : {'linear', 'standard', 'exponential', 'logarithmic'}, optional
      Temperature reduction strategy, by default 'linear'.
      - 'linear': Linear decrease from T_start to T_end
      - 'standard': Geometric cooling (T *= alpha)
      - 'exponential': Exponential decay (T_start * alpha^iteration)
      - 'logarithmic': Logarithmic cooling schedule
  alpha : float, optional
      Cooling parameter, by default 0.1. Usage depends on cooling_schedule.
      - Ignored for 'linear' and 'logarithmic' schedules
  seed : int, optional
      Random seed for reproducible results, by default 0.
  n_cells : int, optional
      Total number of cells in the network scenario, by default 7.
      Only the first 7 (central) cells have variable power levels.
  n_ues : int, optional
      Number of user equipments (UEs) in the scenario, by default 30.
  p_low_dBm : int or float, optional
      Lower bound for transmit power levels in dBm, by default 40.
  p_high_dBm : int or float, optional
      Upper bound for transmit power levels in dBm, by default 56.
  step_dBm : int or float, optional
      Step size for power level increments in dBm, by default 3.
      Valid power levels are generated as range(p_low_dBm, p_high_dBm+1, step_dBm).
  fixed_power_cells_config_dBm : list of float, optional
      Power configuration for fixed power cells in dBm, by default [-np.inf]*12.
      Length should match the number of non-variable cells in the network.
      -np.inf represents disabled/minimum power cells.
  outer_cells_power_dBm : float or int, optional
      Power level for outer cells in dBm, by default 31.
      Applied to cells specified in outer_cells parameter.
  outer_cells : list of int, optional
      Indices of outer cells that use fixed power levels, by default [10,14,18].
      These cells are not optimized and maintain constant power.
  verbose : bool, optional
      If True, print detailed progress information during optimization,
      by default False.
  
  Returns
  -------
  dict
      Dictionary containing optimization results for each iteration with structure:
      {iteration_number: {
          'POWER_COMBINATION': ndarray,
          'SUM_THROUGHPUT': float,
          'SCORE': float,
          'TEMP': float,
          'P_ACCEPT': float,
          'RAND_NUM': float,
          'N_JUMPS': int,
          'ACCEPT_REASON': str
      }}
      
      Where:
      - iteration_number : int
          Iteration index (0-based)
      - POWER_COMBINATION : ndarray
          Candidate or accepted power configuration for central cells
      - SUM_THROUGHPUT : float
          Total system throughput in Mbps
      - SCORE : float
          Energy efficiency score (Mbps/Joule)
      - TEMP : float
          Current temperature value
      - P_ACCEPT : float
          Acceptance probability for the candidate solution
      - RAND_NUM : float
          Random number used in probabilistic acceptance
      - N_JUMPS : int
          Cumulative count of accepted worse solutions
      - ACCEPT_REASON : str
          Reason for accepting/rejecting the candidate solution
  
  Notes
  -----
  The algorithm follows these steps:
  1. Initialize with random power configuration for central cells
  2. For each iteration:
      a. Generate candidate by modifying one random cell's power
      b. Calculate energy efficiency score using f_tp_power_energy_score
      c. Accept if better, or probabilistically if worse based on temperature
      d. Update temperature according to cooling schedule
  3. Record all evaluations and decisions in results dictionary
  
  The acceptance probability for worse solutions follows the Boltzmann
  distribution: P = exp(Δscore / temperature), where Δscore is the
  difference in energy efficiency scores.
  
  Valid power levels include -np.inf (i.e. cell is powered OFF) plus the range
  [p_low_dBm, p_high_dBm] with step_dBm increments.
  
  
  References
  ----------
  .. [1] Kirkpatrick, S., Gelatt Jr, C. D., & Vecchi, M. P. (1983). 
          Optimization by simulated annealing. Science, 220(4598), 671-680.
  .. [2] Černý, V. (1985). Thermodynamical approach to the traveling
          salesman problem: An efficient simulation algorithm.
          Journal of Optimization Theory and Applications, 45(1), 41-51.
  """
      
  crm=get_scenario(seed=seed, n_cells=n_cells, n_ues=n_ues, bw_MHz=10.0)
  crm_SA=get_CRM_SA(crm)
  valid_power_levels=[-np.inf, *np.arange(p_low_dBm, p_high_dBm+1, step_dBm)]

  simulated_annealing_results={}
  i=0
  if verbose:
    print(f'Starting EVAL{i:03}...')
  initial_solution=crm_SA.rng.choice(valid_power_levels, size=7)  # Size is 7 for the central 7 cells.

  if verbose:
    print(f'Initial TX powers:\t{initial_solution}')
  f_accepted_solution=f_tp_power_energy_score(x=initial_solution, crm_SA=crm_SA, fixed_power_cells_config_dBm=fixed_power_cells_config_dBm, outer_cells_power_dBm=outer_cells_power_dBm, outer_cells=outer_cells)
  max_nit=max_n_iterations
  accepted_solution=initial_solution.copy()
  temperature=T_start
  acceptance_probability=1.0
  random_number=0.0
  n_accepted_worse_solutions=0

  # Record the initial solution and score
  simulated_annealing_results[i]={
      'POWER_COMBINATION': accepted_solution,
      'SUM_THROUGHPUT'   : f_accepted_solution[0],
      'SCORE'            : f_accepted_solution[-1],
      'TEMP'             : temperature,
      'P_ACCEPT'         : acceptance_probability,
      'RAND_NUM'         : random_number,
      'N_JUMPS'          : n_accepted_worse_solutions,
      'ACCEPT_REASON'    : 'INITIAL SOLUTION'
    }

  while max_nit>1:
    i+=1
    max_nit-=1

    # Select a random cell
    rand_cell = crm_SA.rng.choice(a=len(initial_solution))
    # Select a random transmit power
    rand_tx_power=crm_SA.rng.choice(a=valid_power_levels)
    # Change the transmit power of the random cell
    candidate_solution=accepted_solution.copy()
    candidate_solution[rand_cell]=rand_tx_power
    # Calculate the candidate solution scores
    f_candidate_solution=f_tp_power_energy_score(x=candidate_solution, crm_SA=crm_SA, fixed_power_cells_config_dBm=fixed_power_cells_config_dBm, outer_cells_power_dBm=outer_cells_power_dBm, outer_cells=outer_cells)
    # Calculate the difference between the current solution and candidate solution score.
    delta_score=f_candidate_solution[-1]-f_accepted_solution[-1]
    if np.greater_equal(delta_score, 0.0):
      accept_reason='BETTER SOLUTION FOUND'
      # This means we have a better score, so we accept it.
      if verbose:
        print(f'Solution accepted...')
        print(f'Acceptance reason: BETTER SOLUTION FOUND.')
        print(f'delta: {delta_score}')
        print(f'\tTP\tPower\tES\tCombination\nOLD:\t{f_accepted_solution[0]:.2f}\t{f_accepted_solution[1]:.2f}\t{f_accepted_solution[2]:.2f}\t{accepted_solution}') # FOR DEBUGGING
        print(f'NEW:\t{f_candidate_solution[0]:.2f}\t{f_candidate_solution[1]:.2f}\t{f_candidate_solution[2]:.2f}\t{candidate_solution}')
      acceptance_probability=1.0
      random_number=0.0
      f_accepted_solution=f_candidate_solution
      accepted_solution=candidate_solution.copy()
    else:
      acceptance_probability=np.exp(delta_score/temperature)
      random_number=crm_SA.rng.random()
      if np.less(random_number, acceptance_probability): # I *THINK* this stays as 'less than' at high temperatures.
        accept_reason = f'Random jump: {random_number:.4f} < {acceptance_probability:.4f}'
        if verbose:
          print(f'Solution accepted...')
          print(f'Acceptance reason: RAND_NUM < P_ACCEPT ({random_number} < {acceptance_probability}).')
          print(f'\tTP\tPower\tES\tCombination\nOLD:\t{f_accepted_solution[0]:.2f}\t{f_accepted_solution[1]:.2f}\t{f_accepted_solution[2]:.2f}\t{accepted_solution}') # FOR DEBUGGING
          print(f'NEW:\t{f_candidate_solution[0]:.2f}\t{f_candidate_solution[1]:.2f}\t{f_candidate_solution[2]:.2f}\t{candidate_solution}')
        n_accepted_worse_solutions+=1
        f_accepted_solution=f_candidate_solution
        accepted_solution=candidate_solution.copy()
      else:
        accept_reason='REJECTED CANDIDATE'
        acceptance_probability=0.0
        random_number=0.0
    simulated_annealing_results[i]={
      'POWER_COMBINATION': candidate_solution,
      'SUM_THROUGHPUT'   : f_candidate_solution[0],
      'SCORE'            : f_candidate_solution[-1],
      'TEMP'             : temperature,
      'P_ACCEPT'         : acceptance_probability,
      'RAND_NUM'         : random_number,
      'N_JUMPS'          : n_accepted_worse_solutions,
      'ACCEPT_REASON'    : accept_reason
    }
    
    # temperature*=alpha
    if cooling_schedule=='standard':
      temperature=standard_cooling(temperature, alpha)
    if cooling_schedule=='linear':
      temperature=linear_cooling(T_start, T_end, max_n_iterations, i)
    if cooling_schedule=='exponential':
      temperature=exponential_cooling(T_start, alpha, i)
    if cooling_schedule=='logarithmic':
      temperature=logarithmic_cooling(T_start, T_end, max_n_iterations, i)

  if verbose:  
    print(f'EVAL{i:03} RESULTS: INITIAL={initial_solution}, FINAL={accepted_solution}, SCORE={f_accepted_solution[-1]:.3f}, N_JUMPS={n_accepted_worse_solutions}.')
  return simulated_annealing_results

def run_vanilla_hillclimb(seed=0, 
                          n_cells=0, 
                          n_ues=0, 
                          p_low_dBm=0, 
                          p_high_dBm=0, 
                          step_dBm=0,
                          fixed_power_cells_config_dBm=[-np.inf]*12,
                          outer_cells_power_dBm=0,
                          outer_cells=[10,14,18],
                          out_dir=''):
  """
  Execute vanilla hill climbing optimization with result logging and CSV export.
  
  This function executes the vanilla hill climbing algorithm for cellular power 
  allocation optimization, handles timing and logging, and exports results to 
  a timestamped CSV file.
  
  Parameters
  ----------
  seed : int, optional
      Random seed for reproducible results, by default 0.
  n_cells : int, optional
      Number of central cells in the network scenario, by default 0.
  n_ues : int, optional
      Number of user equipments (UEs) in the scenario, by default 0.
  p_low_dBm : int or float, optional
      Lower bound for transmit power levels in dBm, by default 0.
  p_high_dBm : int or float, optional
      Upper bound for transmit power levels in dBm, by default 0.
  step_dBm : int or float, optional
      Step size for power level increments in dBm, by default 0.
  fixed_power_cells_config_dBm : list of float, optional
      Power configuration for fixed power cells in dBm, by default [-np.inf]*12.
      Cells with -np.inf are effectively disabled.
  outer_cells_power_dBm : int or float, optional
      Power level for outer boundary cells in dBm, by default 0.
  outer_cells : list of int, optional
      Indices of outer cells with fixed power levels, by default [10,14,18].
  out_dir : str, optional
      Output directory path for result files, by default ''.
  
  Returns
  -------
  int
      Status code indicating successful completion (always returns 0).
  
  Notes
  -----
  The function performs these operations:
  1. Executes vanilla_hillclimb with provided parameters
  2. Records execution time and logs to configured log file
  3. Exports results to timestamped CSV with columns: Iteration, 
      Combination, Throughput (Mbps), Power (Watts), Energy Score (Mbps/Joule)
  
  Output filename format:
  `YYYY-MM-DD-vanilla_hillclimb-{n_cells}cells_{n_ues}ues_p{p_low_dBm}-{p_high_dBm}-ss{step_dBm_str}_seed{seed:03}.csv`
  """
  print(f'RUNNING: SEED{seed:04}')
  t0 = time.time()
  results = vanilla_hillclimb(seed=seed, 
                              n_cells=n_cells, 
                              n_ues=n_ues, 
                              p_low_dBm=p_low_dBm, 
                              p_high_dBm=p_high_dBm, 
                              step_dBm=step_dBm,
                              fixed_power_cells_config_dBm=fixed_power_cells_config_dBm,
                              outer_cells_power_dBm=outer_cells_power_dBm,
                              outer_cells=outer_cells)
  t1 = time.time()
  logging.info(f"Total time taken: {t1 - t0} seconds")
  date_str=dt.now().strftime('%Y-%m-%d')
  step_dBm_str=f'{step_dBm:02}'.replace('.', '-')
  results_fn=f'{date_str}-vanilla_hillclimb-{n_cells}cells_{n_ues}ues_p{p_low_dBm}-{p_high_dBm}-ss{step_dBm_str}_seed{seed:03}.csv'
  results_path=f'./{out_dir}/{results_fn}'
  with open(results_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Iteration','Combination', 'Attached UEs Sum Throughput (Mbps)', 'Power Consumption (Watts)', 'Energy Score (Mbps/Joule)'])
    for iteration, evaulations in results.items():
      for combination, results in evaulations.items():
        writer.writerow([iteration, str(combination), results[0], results[1], results[2]])
  logging.info(f"Results saved to {results_fn}")
  print(f'COMPLETED: SEED {seed:04}')
  return 0


def run_random_restart_hillclimb(n_evals=2,
                                 max_iterations=100,
                                 seed=0, 
                                 n_cells=0, 
                                 n_ues=0, 
                                 p_low_dBm=0, 
                                 p_high_dBm=0, 
                                 step_dBm=0,
                                 fixed_power_cells_config_dBm=[-np.inf]*12,
                                 outer_cells_power_dBm=0,
                                 outer_cells=[10,14,18],
                                 out_dir=''):
  """
Execute random restart hill climbing optimization with result logging and CSV export.

This function executes the random restart hill climbing algorithm for cellular 
power allocation optimization, handles timing and logging, and exports results 
to a timestamped CSV file.

Parameters
----------
n_evals : int, optional
    Number of evaluation rounds (restarts) to perform, by default 2.
max_iterations : int, optional
    Maximum number of global iterations across all restarts, by default 100.
seed : int, optional
    Random seed for reproducible results, by default 0.
n_cells : int, optional
    Number of central cells in the network scenario, by default 0.
n_ues : int, optional
    Number of user equipments (UEs) in the scenario, by default 0.
p_low_dBm : int or float, optional
    Lower bound for transmit power levels in dBm, by default 0.
p_high_dBm : int or float, optional
    Upper bound for transmit power levels in dBm, by default 0.
step_dBm : int or float, optional
    Step size for power level increments in dBm, by default 0.
fixed_power_cells_config_dBm : list of float, optional
    Power configuration for fixed power cells in dBm, by default [-np.inf]*12.
    Cells with -np.inf are effectively disabled.
outer_cells_power_dBm : int or float, optional
    Power level for outer boundary cells in dBm, by default 0.
outer_cells : list of int, optional
    Indices of outer cells with fixed power levels, by default [10,14,18].
out_dir : str, optional
    Output directory path for result files, by default ''.

Returns
-------
int
    Status code indicating successful completion (always returns 0).

Notes
-----
The function performs these operations:
1. Executes random_restart_hillclimb with provided parameters
2. Records execution time and logs to configured log file
3. Exports results to timestamped CSV with columns: Global_iteration, 
    Evaluation, Iteration, Combination, Throughput (Mbps), Power (Watts), 
    Energy Score (Mbps/Joule)

Output filename format:
`YYYY-MM-DD-random_restart_hillclimb-{n_cells}cells_{n_ues}ues_p{p_low_dBm}-{p_high_dBm}-ss{step_dBm}_nevals{n_evals:04}_maxiterations{max_iterations:06}_seed{seed:03}.csv`
"""
  print(f'RUNNING: SEED{seed:04}')
  t0 = time.time()
  results = random_restart_hillclimb(n_evals=n_evals,
                                     max_iterations=max_iterations,
                                     seed=seed, 
                                     n_cells=n_cells, 
                                     n_ues=n_ues, 
                                     p_low_dBm=p_low_dBm, 
                                     p_high_dBm=p_high_dBm, 
                                     step_dBm=step_dBm,
                                     fixed_power_cells_config_dBm=fixed_power_cells_config_dBm,
                                     outer_cells_power_dBm=outer_cells_power_dBm,
                                     outer_cells=outer_cells)
  t1 = time.time()
  logging.info(f"Total time taken: {t1 - t0} seconds")
  date_str=dt.now().strftime('%Y-%m-%d')
  step_dBm_str=f'{step_dBm:02}'.replace('.', '-')
  results_fn=f'{date_str}-random_restart_hillclimb-{n_cells}cells_{n_ues}ues_p{p_low_dBm}-{p_high_dBm}-ss{step_dBm_str}_nevals{n_evals:04}_maxiterations{max_iterations:06}_seed{seed:03}.csv'
  results_path=f'./{out_dir}/{results_fn}'
  with open(results_path, mode='w', newline='') as file:
    g_it=0
    writer = csv.writer(file)
    writer.writerow(['Global_iteration','Evaluation','Iteration','Combination', 'Attached UEs Sum Throughput (Mbps)', 'Power Consumption (Watts)', 'Energy Score (Mbps/Joule)'])

    for evaluation, iterations in results.items():
      for iteration, combinations in iterations.items():
        for combination, results in combinations.items():
          writer.writerow([g_it, evaluation, iteration, str(combination), results[0], results[1], results[2]])
          g_it+=1
  logging.info(f"Results saved to {results_fn}")
  print(f'COMPLETED: SEED {seed:04}')
  return 0

def run_simulated_annealing(max_n_iterations=100,
                            T_start=1000, # In Kelvin
                            T_end=0.1,
                            cooling_schedule='linear',
                            alpha=0.5,
                            seed=0, 
                            n_cells=7, 
                            n_ues=30, 
                            p_low_dBm=40, 
                            p_high_dBm=56, 
                            step_dBm=3,
                            fixed_power_cells_config_dBm=[-np.inf]*12,
                            outer_cells_power_dBm=31,
                            outer_cells=[10,14,18],
                            out_dir=''):  
  """
Execute simulated annealing optimization with result logging and CSV export.

This function executes the simulated annealing algorithm for cellular power 
allocation optimization, handles timing and logging, and exports results to 
a timestamped CSV file.

Parameters
----------
max_n_iterations : int, optional
    Maximum number of iterations to run the algorithm, by default 100.
T_start : float, optional
    Initial temperature in Kelvin for the annealing process, by default 1000.
T_end : float, optional
    Final temperature in Kelvin, by default 0.1.
    Used only for linear and logarithmic cooling schedules.
cooling_schedule : {'linear', 'standard', 'exponential', 'logarithmic'}, optional
    Temperature reduction strategy, by default 'linear'.
alpha : float, optional
    Cooling parameter, by default 0.5. Usage depends on cooling_schedule.
seed : int, optional
    Random seed for reproducible results, by default 0.
n_cells : int, optional
    Number of central cells in the network scenario, by default 7.
n_ues : int, optional
    Number of user equipments (UEs) in the scenario, by default 30.
p_low_dBm : int or float, optional
    Lower bound for transmit power levels in dBm, by default 40.
p_high_dBm : int or float, optional
    Upper bound for transmit power levels in dBm, by default 56.
step_dBm : int or float, optional
    Step size for power level increments in dBm, by default 3.
fixed_power_cells_config_dBm : list of float, optional
    Power configuration for fixed power cells in dBm, by default [-np.inf]*12.
    Cells with -np.inf are effectively disabled (i.e. powered OFF).
outer_cells_power_dBm : int or float, optional
    Power level for outer boundary cells in dBm, by default 31.
outer_cells : list of int, optional
    Indices of outer cells with fixed power levels, by default [10,14,18].
out_dir : str, optional
    Output directory path for result files, by default ''.

Returns
-------
int
    Status code indicating successful completion (always returns 0).

Output filename format:
`YYYY-MM-DD-simulated_annealing-{n_cells}cells_{n_ues}ues_p{p_low_dBm}-{p_high_dBm}-ss{step_dBm_str}_seed{seed:03}_maxiterations{max_n_iterations:06}.csv`
"""
  print(f'RUNNING: SEED{seed:04}')
  t0 = time.time()
  results = simulated_annealing(max_n_iterations=max_n_iterations,
                                T_start=T_start, # In Kelvin
                                T_end=T_end,
                                cooling_schedule=cooling_schedule,
                                alpha=alpha,
                                seed=seed, 
                                n_cells=n_cells, 
                                n_ues=n_ues, 
                                p_low_dBm=p_low_dBm, 
                                p_high_dBm=p_high_dBm, 
                                step_dBm=step_dBm,
                                fixed_power_cells_config_dBm=fixed_power_cells_config_dBm,
                                outer_cells_power_dBm=outer_cells_power_dBm,
                                outer_cells=outer_cells,
                                )
  t1 = time.time()
  logging.info(f"Total time taken: {t1 - t0} seconds")
  date_str=dt.now().strftime('%Y-%m-%d')
  step_dBm_str=f'{step_dBm:02}'.replace('.', '-')
  results_fn=f'{date_str}-simulated_annealing-{n_cells}cells_{n_ues}ues_p{p_low_dBm}-{p_high_dBm}-ss{step_dBm_str}_seed{seed:03}_maxiterations{max_n_iterations:06}.csv'
  results_path=f'./{out_dir}/{results_fn}'
  with open(results_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Iteration','Candidate Solution', 'Energy Score (Mbps/Joule)', 'Temperature (K)', 'P_ACCEPT', 'Random_number', 'N_jumps', 'ACCEPT_REASON'])
    for iteration, kvs in results.items():
      writer.writerow([iteration, str(kvs['POWER_COMBINATION']), kvs['SCORE'], kvs['TEMP'], kvs['P_ACCEPT'], kvs['RAND_NUM'], kvs['N_JUMPS'], kvs['ACCEPT_REASON']])
  logging.info(f"Results saved to {results_fn}")
  print(f'COMPLETED: SEED {seed:04}')
  return 0

  
  

if __name__ == '__main__':
  pass
