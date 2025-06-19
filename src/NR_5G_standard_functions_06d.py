# Kishan Sthankiya 2024-11-29 fix TBS lookup for Ninfo<3824 in get_3GPP_throughput_Mbps_vect
# Kishan Sthankiya 2024-10-16
# addition of transport block size table (TS 138 214 V18.4.0 (2024-10) 'Table 5.1.3.2-1' )
# Keith Briggs 2020-10-13
# Keith Briggs 2020-09-21
# map rsrp to reported value: 5G in Bullets p389 Table 237
# map SINR to CQI: https://uk.mathworks.com/help/5g/ug/5g-nr-cqi-reporting.html
# TODO: Future work may look to map SINR to effective SINR (e.g. https://arxiv.org/pdf/2001.10309 - REMARK 1)

import sys
import numpy as np
import pandas as pd
from math import ceil,log2
from functools import lru_cache
from dataclasses import dataclass
from matplotlib import pyplot as plt

# BELOW: Slight kludge to import from src folder while in test folder
sys.path.append('../src')

from color_text_00 import Color_text
from fig_timestamp_01 import fig_timestamp
from dB_conversions_01 import ratio_to_dB

__version__='06d'
color_text=Color_text()


# LOOKUP TABLES

SINR90PC=np.array([-float('inf'),-1.89,-0.82,0.95,2.95,4.90,7.39,8.89,11.02,13.32,14.68,16.62,18.91,21.58,24.88,29.32,float('inf'),])

# TS_38_214.pdf page 43 Table 5.2.2.1-2: 4-bit CQI Table for reporting CQI based on QPSK
# The CQI indices and their interpretations are given in Table 5.2.2.1-2 or Table 5.2.2.1-4 for reporting CQI based on QPSK, 16QAM and 64QAM. The CQI indices and their interpretations are given in Table 5.2.2.1-3 for reporting CQI based on QPSK, 16QAM, 64QAM and 256QAM.
_CQI_TABLE_1=np.array([
[ 0,  -float('inf'), -float('inf'),-float('inf')],
[ 1,  2,   78,  0.1523],
[ 2,  2,  120,  0.2344],
[ 3,  2,  193,  0.3770],
[ 4,  2,  308,  0.6016],
[ 5,  2,  449,  0.8770],
[ 6,  2,  602,  1.1758],
[ 7,  4,  378,  1.4766],
[ 8,  4,  490,  1.9141],
[ 9,  4,  616,  2.4063],
[10,  6,  466,  2.7305],
[11,  6,  567,  3.3223],
[12,  6,  666,  3.9023],
[13,  6,  772,  4.5234],
[14,  6,  873,  5.1152],
[15,  6,  948,  5.5547],
[16,  float('inf'), float('inf'),float('inf')],
])

# 38.214 Table 5.1.3.2-2
# http://www.techplayon.com/5g-nr-modulation-and-coding-scheme-modulation-and-code-rate/
# MCS Index: & Modulation Order Qm & Target code Rate x1024 R & Spectral efficiency\\
MCS_TABLE_1_FOR_PDSCH={
 0: ( 2,120,0.2344),
 1: ( 2,157,0.3066),
 2: ( 2,193,0.3770),
 3: ( 2,251,0.4902),
 4: ( 2,308,0.6016),
 5: ( 2,379,0.7402),
 6: ( 2,449,0.8770),
 7: ( 2,526,1.0273),
 8: ( 2,602,1.1758),
 9: ( 2,679,1.3262),
10: ( 4,340,1.3281),
11: ( 4,378,1.4766),
12: ( 4,434,1.6953),
13: ( 4,490,1.9141),
14: ( 4,553,2.1602),
15: ( 4,616,2.4063),
16: ( 4,658,2.5703),
17: ( 6,438,2.5664),
18: ( 6,466,2.7305),
19: ( 6,517,3.0293),
20: ( 6,567,3.3223),
21: ( 6,616,3.6094),
22: ( 6,666,3.9023),
23: ( 6,719,4.2129),
24: ( 6,772,4.5234),
25: ( 6,822,4.8164),
26: ( 6,873,5.1152),
27: ( 6,910,5.3320),
28: ( 6,948,5.5547),
# 29: ( 2,'reserved', 'reserved'),
# 30: ( 4,'reserved', 'reserved'),
# 31: ( 6,'reserved', 'reserved'),
}

MCS_TABLE_1_FOR_PDSCH_ARRAY=np.array([[key, *value] for key, value in MCS_TABLE_1_FOR_PDSCH.items()])

# 38.214 (V18.4.0) Table 5.1.3.2-1: Transport block size for N_info ≤ 3824 bits
TBS_TABLE_3824=np.array([
  [1,24],[2,32],[3,40],[4,48],[5,56],[6,64],[7,72],[8,80],[9,88],[10,96],[11,104],[12,112],[13,120],[14,128],[15,136],[16,144],[17,152],[18,160],[19,168],[20,176],[21,184],[22,192],[23,208],[24,224],[25,240],[26,256],[27,272],[28,288],[29,304],[30,320],[31,336],[32,352],[33,368],[34,384],[35,408],[36,432],[37,456],[38,480],[39,504],[40,528],[41,552],[42,576],[43,608],[44,640],[45,672],[46,704],[47,736],[48,768],[49,808],[50,848],[51,888],[52,928],[53,984],[54,1032],[55,1064],[56,1128],[57,1160],[58,1192],[59,1224],[60,1256],[61,1288],[62,1320],[63,1352],[64,1416],[65,1480],[66,1544],[67,1608],[68,1672],[69,1736],[70,1800],[71,1864],[72,1928],[73,2024],[74,2088],[75,2152],[76,2216],[77,2280],[78,2408],[79,2472],[80,2536],[81,2600],[82,2664],[83,2728],[84,2792],[85,2856],[86,2976],[87,3104],[88,3240],[89,3368],[90,3496],[91,3624],[92,3752],[93,3824],
  ])

NPRB_TABLE_DICT = {'Numerology': {0: 0, 1: 1, 2: 2}, 
                   'SCS_kHz': {0: 15, 1: 30, 2: 60}, 
                   'BW_5_MHz_nPRBs': {0: 25.0, 1: 11.0, 2: np.nan}, 
                   'BW_10_MHz_nPRBs': {0: 52, 1: 24, 2: 11}, 
                   'BW_15_MHz_nPRBs': {0: 79, 1: 38, 2: 18}, 
                   'BW_20_MHz_nPRBs': {0: 106, 1: 51, 2: 24}, 
                   'BW_25_MHz_nPRBs': {0: 133, 1: 65, 2: 31}, 
                   'BW_30_MHz_nPRBs': {0: 160, 1: 78, 2: 38}, 
                   'BW_35_MHz_nPRBs': {0: 188, 1: 92, 2: 44}, 
                   'BW_40_MHz_nPRBs': {0: 216, 1: 106, 2: 51}, 
                   'BW_45_MHz_nPRBs': {0: 242, 1: 119, 2: 58}, 
                   'BW_50_MHz_nPRBs': {0: 270, 1: 133, 2: 65}, 
                   'BW_60_MHz_nPRBs': {0: np.nan, 1: 162.0, 2: 79.0}, 'BW_70_MHz_nPRBs': {0: np.nan, 1: 189.0, 2: 93.0}, 'BW_80_MHz_nPRBs': {0: np.nan, 1: 217.0, 2: 107.0}, 'BW_90_MHz_nPRBs': {0: np.nan, 1: 245.0, 2: 121.0}, 'BW_100_MHz_nPRBs': {0: np.nan, 1: 273.0, 2: 135.0}}

NPRB_TABLE=np.array([
  [  5,     25,  11, np.nan],
  [ 10,     52,  24,     11],
  [ 15,     79,  38,     18],
  [ 20,    106,  51,     24],
  [ 25,    133,  65,     31],
  [ 30,    160,  78,     38],
  [ 35,    188,  92,     44],
  [ 40,    216, 106,     51],
  [ 45,    242, 119,     58],
  [ 50,    270, 133,     65],
  [ 60, np.nan, 162,     79],
  [ 70, np.nan, 189,     93],
  [ 80, np.nan, 217,    107],
  [ 90, np.nan, 245,    121],
  [100, np.nan, 273,    135],
])

NUMEROLOGY_TABLE = np.array([
  # Numerology, subcarrier spacing (kHz), supported for data (PDSCH/PUSCH) 0=False 1=true, symbol duration (us), slot duration (ms), symbols per slot, slots per frame, subcarriers per PRB, min PRB, max PRB, min BW (MHz), max BW (MHz)
  # FIXME: min and max PRB don't look correct.
  [0,  15, 1, 66.67, 1.0,    14,  10, 12, 20, 275,],
  [1,  30, 1, 33.33, 0.5,    14,  20, 12, 20, 275,],
  [2,  60, 1, 16.67, 0.25,   14,  40, 12, 20, 275,],
  [3, 120, 1,  8.33, 0.125,  14,  80, 12, 20, 275,],
  [4, 240, 0,  4.17, 0.0625, 14, 160, 12, 20, 138,]
])

# FUNCTIONS by Kishan Sthankiya

def CQI_to_64QAM_efficiency_vect(cqi_array):
  # Ensure CQI is clipped within the valid range (0 to 15)
  cqi_array = np.clip(cqi_array, 0, 15)

  # Calculate the MCS indices based on mapping
  CQI_to_MCS= np.clip(np.floor(28 * cqi_array / 15.0).astype(int), 0, 28)

  # Retrieve spectrral efficiency values from the MCS table
  se_array = MCS_TABLE_1_FOR_PDSCH_ARRAY[CQI_to_MCS][:,-1]

  return CQI_to_MCS, se_array

def get_nPRB(s):
  ''' Get the nPRB value based on the serving cell's bandwidth and numerology value.'''
  col=f'BW_{int(s.serving_cell.bw_MHz)}_MHz_nPRBs'
  nPRB = NPRB_TABLE[(NPRB_TABLE['Numerology']==s.mu) & (NPRB_TABLE.loc[:,col]>0)][col].values[0]
  # FIXME: maybe put into a dictionary instead.
  x=nPRB*s.serving_cell.subband_mask/s.serving_cell.get_nattached()
  return int(np.sum(x))

def get_nPRB_array(bw_MHz_array, mu_array):
  bw_indices = (bw_MHz_array.astype(int) // 5) - 1
  nPRB_array = NPRB_TABLE[bw_indices, mu_array+1] # Fancy indexing
  return nPRB_array

def get_slot_size_ms(numerology):
  return 1/(2**numerology)

def get_numerology(fc_GHz):
  if fc_GHz < 6.0:
    return 0,1
  elif fc_GHz > 24.5:
    return 3
  else:
    return 2

def sinr_to_se(sinr_matrix, bw_Hz):
  # Take a numpy matrix and return Shannon-based spectral efficiency (assuming a bandwidth of 1 Hz).
  return bw_Hz*np.log1p(sinr_matrix)/np.log(2)

def sinr_to_cqi(sinr_dB_matrix):
  # Take a numpy matrix (M x N) and return a matrix of CQI values with the same shape.
  return np.searchsorted(SINR90PC,sinr_dB_matrix,side='right')-1

def se_to_cqi(se_matrix, attachment_matrix=None):
  # Using the logic from the slide set we find the position the SE value fits in _CQI_Table_1
  if isinstance(attachment_matrix, np.ndarray):
    se_array = se_matrix[attachment_matrix]
    return np.searchsorted(_CQI_TABLE_1[:,-1],se_array)-1
  else:
    return np.searchsorted(_CQI_TABLE_1[:,-1],se_matrix)-1

def se_to_mcs(se_matrix, attachment_matrix=None):
  if isinstance(attachment_matrix, np.ndarray):
    se = se_matrix[attachment_matrix]
  else:
    se = se_matrix

  # We'll need the CQI values a little later
  cqi_idx = se_to_cqi(se_matrix, attachment_matrix=attachment_matrix)

  # Get the MCS index matrix
  mcs_idx = np.searchsorted(MCS_TABLE_1_FOR_PDSCH_ARRAY[:,-1],se)-1

  return mcs_idx

def get_Qm_R_from_CQI_MCS(cqi_idx, mcs_idx):
  # What is the Qm from the CQI table?  
  cqi_Qm_array = _CQI_TABLE_1[:,1]
  cqi_Qm = np.take(cqi_Qm_array, cqi_idx)

  # What is the Qm from the MCS table?
  mcs_Qm_array = MCS_TABLE_1_FOR_PDSCH_ARRAY[:,1]
  mcs_Qm = np.take(mcs_Qm_array, mcs_idx)

  # Is the Qm from the CQI table the same as the Qm from the MCS table?
  Qm_match = cqi_Qm == mcs_Qm
  # TODO: What to do if Qm doesn't match?

  # What is the R from the CQI table?
  cqi_R_array = _CQI_TABLE_1[:,2]
  cqi_R = np.take(cqi_R_array, cqi_idx)

  # What is the R from the MCS table?
  mcs_R_array = MCS_TABLE_1_FOR_PDSCH_ARRAY[:,2]
  mcs_R = np.take(mcs_R_array, mcs_idx)

  # Is the R from the CQI table the same as the R from the MCS table?
  R_match = cqi_R == mcs_R

  # Test if the Qm match is False AND the R match is False
  Qm_and_R_match = Qm_match + R_match

  # TODO: In the case where Qm_and_R_match is False, what should we do?

  # In the case where the CQI index is 0, the Qm and R values are -inf
  Qm_R = np.where(cqi_idx==0, -np.inf, (mcs_Qm, mcs_R))

  return Qm_R

   
def sinr_to_mcs(sinr):
  # Take an array of SINR values and return an array of MCS values.
  se_bps_Hz = np.log1p(sinr)/np.log(2)
  mcs_idx = np.searchsorted(MCS_TABLE_1_FOR_PDSCH_ARRAY[:,-1],se_bps_Hz)-1
  return mcs_idx

def mcs_to_se(mcs_idx):
  # Take an array of MCS values and return an array of spectral efficiency values.
  se_bps_Hz = np.take(MCS_TABLE_1_FOR_PDSCH_ARRAY[:,-1],mcs_idx)
  return se_bps_Hz
  

def get_simple_data_rate_Mbps(radio_state):
  r=radio_state
  # Check if any parameter in radio_state is None
  if any(param is None for param in [r.NRB_sc, r.Nsh_symb, r.NPRB_oh, r.nPRB, r.Qm, r.v, r.R]):
      return 0.0, 0.0
  T_mu_s = 1e-3/(14*(2**r.mu))
  rate = 1e-6*(r.v*r.Qm*(r.R/1024)*((r.nPRB*12)/T_mu_s)*(1-r.NPRB_oh))
  if r.operating_mode == 'FDD':
    dl_rate_Mbps = rate
    ul_rate_Mbps = rate
  dl_rate_Mbps = rate * r.dl_ul_ratio
  ul_rate_Mbps = rate * (1-r.dl_ul_ratio)
  return dl_rate_Mbps, ul_rate_Mbps

def get_3GPP_data_rate_Mbps_vect_simple(rsm):
  '''
  Taken from the 3GPP 38.306 V18.3.0 (2024-10) Section 4.1.2.
  https://www.etsi.org/deliver/etsi_ts/138300_138399/138306/18.03.00_60/ts_138306v180300p.pdf
  '''
  throughput_Mbps= np.zeros_like(rsm['cqi'])
  valid_cqi_mask = rsm['cqi'] > 0
  T_mu_s = 1e-3/(14*(2**rsm['mu_numerology']))
  rate = 1e-6*(rsm['v_layers']*rsm['modulation']*(rsm['code_rate']/1024))*((rsm['UE_nPRB_allocation']*12)/T_mu_s)*(1-rsm['NPRB_oh'])
  return rate

def get_3GPP_throughput_Mbps_vect_V2(rsm):
  # rsm is a dictionary.

  # We'll start with an array of NaNs for the throughput values
  throughput_Mbps = np.full(rsm['cqi'].shape, np.nan)

  # For any cqi value less than 1, the throughput is 0.0
  np.where(rsm['cqi'] < 1, 0.0, throughput_Mbps)

  # For any cqi value greater than 0, we'll calculate the throughput.
  # But first we'll need a mask for the valid cqi values.
  valid_cqi_mask = rsm['cqi'] > 0

  # Next, we'll calculate the transport block size.

  # Start by calculating the no. of PRBs used for overhead.
  NPRB_oh = rsm['NPRB_oh']*rsm['NRB_sc']*rsm['Nsh_symb']

  # Calculate the no. of REs per PRB
  NRE_prime = (rsm['NRB_sc']*rsm['Nsh_symb']) - NPRB_oh - rsm['NPRB_DMRS']

  # Calculate the no. of REs
  NRE = np.minimum(156, NRE_prime)*rsm['UE_nPRB_allocation']

  # Calculate the no. of information bits
  Ninfo = NRE*rsm['modulation']*rsm['code_rate']*rsm['v_layers']

  # Calculate the TBS for Ninfo <= 3824
  mask_Ninfo_leq_3824 = Ninfo <= 3824
  n = np.maximum(3, np.floor(np.log2(Ninfo, where=(Ninfo>0.0)))-6)
  Ninfo_prime_leq_3824=np.maximum(24, (2**n)*np.floor(Ninfo/(2**n)))
  TBS_array=TBS_TABLE_3824.T[1]
  TBS_bits_leq_3824=TBS_array[np.clip(np.searchsorted(TBS_array,Ninfo_prime_leq_3824),min=0,max=92)] # potential fix
  throughput_Mbps[mask_Ninfo_leq_3824]=1e-6*TBS_bits_leq_3824[mask_Ninfo_leq_3824]

  # Process cases where Ninfo > 3824
  mask_Ninfo_gt_3824=np.greater(Ninfo,3824)
  n=np.floor(np.log2(Ninfo-24, where=(np.greater(Ninfo,24))))-5
  Ninfo_prime_gt_3824=np.maximum(
    3840,
    (np.power(2,n))*np.round((Ninfo-24)/(np.power(2,n)))
  )
  C=np.where(np.less_equal(rsm['code_rate']/1024,0.25), 
             np.ceil((Ninfo_prime_gt_3824+24)/3816),
             np.where(np.greater(Ninfo_prime_gt_3824,8424), 
                      np.ceil((Ninfo_prime_gt_3824+24)/8424),
                      1)
              )
  TBS_bits_gt_3824=8*C*np.ceil((Ninfo_prime_gt_3824+24)/(8*C))-24
  throughput_Mbps[mask_Ninfo_gt_3824] = 1e-6*TBS_bits_gt_3824[mask_Ninfo_gt_3824]

  return throughput_Mbps

def get_3GPP_throughput_Mbps_vect(rsm):
  # Initialise result array with -1
  result = np.full(rsm.shape[1],-1.0)

  # Validate rows that can be <=0  
  valid_leq_zero_rows=[0,8,11]                   #serving_cell,mu,NPRB_oh.
  valid_mask = ~np.isin(np.arange(rsm.shape[0]), valid_leq_zero_rows)
  invalid_cols=np.any(rsm[valid_mask,:] <= 0.0, axis=0)
  result[invalid_cols]=0.0

  # Masks
  mask_for_MCS_0=result==0.0

  # Extract variables
  NRB_sc=rsm[9,:]
  Nsh_symb=rsm[10,:]
  NPRB_DMRS=rsm[15,:] # FIXED: Changed from 14 to 15 [2024-11-28 KS]
  NPRB_oh=rsm[11,:]*NRB_sc*Nsh_symb
  NRE_prime=(NRB_sc*Nsh_symb)-NPRB_oh-NPRB_DMRS
  nPRB=rsm[13,:]
  NRE=np.minimum(156, NRE_prime)*nPRB
  R=rsm[4,:]/1024
  Qm=rsm[3,:]
  v=rsm[7,:]
  Ninfo=NRE*R*Qm*v

  # Process cases where Ninfo <= 3824
  mask_Ninfo_leq_3824=Ninfo<=3824
  n=np.maximum(3, np.floor(np.log2(Ninfo, where=(Ninfo>0.0)))-6)
  Ninfo_prime_leq_3824=np.maximum(24, (2**n)*np.floor(Ninfo/(2**n)))
  TBS_array=TBS_TABLE_3824.T[1]
  TBS_bits_leq_3824=TBS_array[np.clip(np.searchsorted(TBS_array,Ninfo_prime_leq_3824),min=0,max=92)] # potential fix
  result[mask_Ninfo_leq_3824]=1e-3*TBS_bits_leq_3824[mask_Ninfo_leq_3824]

  # Process cases where Ninfo > 3824
  mask_Ninfo_gt_3824=np.greater(Ninfo,3824)
  n=np.floor(np.log2(Ninfo-24, where=(np.greater(Ninfo,24))))-5
  Ninfo_prime_gt_3824=np.maximum(
    3840,
    np.maximum(3840, (np.power(2,n))*np.round((Ninfo-24)/(np.power(2,n))))
  )
  C=np.where(np.less_equal(R,0.25), 
             np.ceil((Ninfo_prime_gt_3824+24)/3816),
             np.where(np.greater(Ninfo_prime_gt_3824,8424), 
                      np.ceil((Ninfo_prime_gt_3824+24)/8424),
                      1)
              )
  TBS_bits_gt_3824=8*C*np.ceil((Ninfo_prime_gt_3824+24)/(8*C))-24
  result[mask_Ninfo_gt_3824] = 1e-3*TBS_bits_gt_3824[mask_Ninfo_gt_3824]

  # Ensure MCS 0 throughput is 0.0
  result[mask_for_MCS_0]=0.0

  return result

def determine_3GPP_Throughput_Mbps(radio_state):
  """
  Calculate the 3GPP throughput in Mbps based on the given radio state parameters.

  This function follows the procedure outlined in 3GPP 38.214 V18.4.0 (2024-10), Section 5.1.3.2, pp.38--42.

  NOTE1:  TS 38.306 version 18.1.0 Release 18, Section 4.1.2, states that when only 1 component carrier is configured, the product of layvers (v)* modulation order (Qm) * scaling factor (f) shoild not be less than 4.

  Parameters:
  radio_state (object): An object containing the following attributes:
    - NRB_sc (int): Number of subcarriers in a resource block (RB).
    - Nsh_symb (int): Number of symbols of the PDSCH allocation within the slot.
    - NPRB_oh (int): Overhead of the PRB.
    - nPRB (int): Number of physical resource blocks.
    - R (float): Code rate.
    - Qm (int): Modulation order.
    - v (int): Number of layers.

  Returns:
  float: The calculated throughput in Mbps. Returns 0.0 if any required parameter is None.
  """
  # Procedure taken from 3GPP 38.214 V18.4.0 (2024-10), Section 5.1.3.2, pp.38--42.
  # Filename: ts_138214v180400p.pdf
  if any(param is None for param in [radio_state.NRB_sc, radio_state.Nsh_symb, radio_state.NPRB_oh, radio_state.nPRB, radio_state.R, radio_state.Qm, radio_state.v]):
      return 0.0
  
  # Enforce NOTE1
  if radio_state.nCC == 1:
    if radio_state.v * radio_state.Qm < 4:
      print(color_text.Red(f'The product of layers (v) * modulation order (Qm) should not be less than 4 when only 1 component carrier is configured.'))

  NRB_sc = radio_state.NRB_sc           # Number of subcarriers (sc) in a resource block (RB)
  Nsh_symb = radio_state.Nsh_symb       # Number of symbols of the PDSCH allocation within the slot
  NPRB_DMRS=_DMRS_RE('type1','A',1,0)   # Number of REs for DM-RS per PRB in the scheduled duration including the overhead of the   
                                        # DM-RS CDM groups without data, as indicated by DCI format 1_1, 1_2 or 1_3 or as described for format 1_0 in Clause 5.1.6.2
  NPRB_oh = radio_state.NPRB_oh*NRB_sc*Nsh_symb
  NREprime = (NRB_sc*Nsh_symb)-NPRB_oh-NPRB_DMRS
  NRE = min(156, NREprime) * radio_state.nPRB
  R=radio_state.R/1024
  Ninfo = NRE * R * radio_state.Qm * radio_state.v
  if Ninfo <= 3824:
    # Step 3
    n = max(3, np.floor(log2(Ninfo))-6)
    Ninfo_prime = max(24, (2**n)*np.floor(Ninfo/(2**n)))
    # Use TBS_table_3824 (Table 5.1.3.2-1) to find the closest TBS that is not less than Ninfo_prime
    TBS_bits = 0
    for i in range(len(TBS_TABLE_3824)):  #FIXME bisect (may need a -inf and +inf entry)
      if TBS_TABLE_3824[i][1] >= Ninfo_prime:
        TBS_bits = TBS_TABLE_3824[i][1]
        return 1e-3*TBS_bits # in Mbps
  else:
    # Step 4
    n = np.floor(log2(Ninfo-24)) - 5
    Ninfo_prime = max(3840, (2**n)*round((Ninfo-24)/(2**n)))
    if R <= 0.25:
      C = np.ceil((Ninfo_prime+24)/3816)
      TBS_bits = 8 * C * np.ceil((Ninfo_prime+24)/(8*C))-24
    else:
      if Ninfo_prime > 8424:
        C = np.ceil((Ninfo_prime+24)/8424)
        TBS_bits = 8*C*np.ceil((Ninfo_prime+24)/(8*C))-24
      else:
        TBS_bits = 8 * np.ceil((Ninfo_prime+24) / 8)-24
    return 1e-3*TBS_bits # in Mbps

def SE_to_CQI_Qm_R_SE(SE, tBLER_10pc=True): # OOP implementation for AIMM. Deprecate? (KS 2024-11-15)
  """Gets the CQI index, modulation order Qm, target code rate x1024 R, and 3GPP Spectral efficiency from the Shannon based SE."""
  # Add a dimension to SE that is the same as the last dimension of _CQI_Table_1
  SE = np.array(SE)[:, np.newaxis]
  if tBLER_10pc:
    sinr_dB = ratio_to_dB(np.power(2, SE) - 1)
    tBLER_cqi_i = SINR_to_CQI(sinr_dB)
    return np.take(_CQI_TABLE_1, tBLER_cqi_i, axis=0) 
  return _CQI_TABLE_1[np.searchsorted(a=_CQI_TABLE_1[:, -1], v=SE, side='left')-1]


def SE_to_MCS_Qm_R_SE(sinr_dB, se_Shannon, cqi_i, cqi_Qm, cqi_R, cqi_SE, t_BLER10pc=True): 
  # OOP implementation for AIMM. Deprecate? (KS 2024-11-15)
  """
  Gets the MCS index, modulation order Qm, and target code rate x1024 R from the spectral efficiency.

  Parameters:
  sinr_dB (float)                   : Signal-to-Interference-plus-Noise Ratio in dB.
  SE_Shannon (float or np.ndarray)  : Spectral Efficiency based on Shannon's capacity.
  cqi_i (int)                       : Channel Quality Indicator index.
  cqi_Qm (int)                      : Modulation order corresponding to the CQI.
  cqi_R (int)                       : Code rate corresponding to the CQI.
  cqi_SE (float)                    : Spectral Efficiency corresponding to the CQI.
  t_BLER10pc (bool)                 : Flag to indicate if 10% BLER should be considered. Default is True.

  """
  if isinstance(se_Shannon, np.ndarray):
    se_Shannon = se_Shannon[0]

  df_mcs = pd.DataFrame(MCS_TABLE_1_FOR_PDSCH).T
  df_mcs.columns = ['Qm', 'R', 'SE']

  if cqi_i != 0:  # If cqi_Qm is not 0
    # Get the rows where the SE is less than or equal to the CQI SE
    df_mcs = df_mcs[df_mcs['SE'] <= cqi_SE]
    # Next get the rows where the Qm is equal to cqi_Qm
    df_mcs = df_mcs[df_mcs['Qm'] == cqi_Qm]
  else:
    return None, None, None, None  # MCS, Qm, R and SE are all N/A
  # Compare the Shannon CQI with the CQI from the 10% BLER lookup
  if t_BLER10pc:
    t_BLER_cqi_i = SINR_to_CQI(sinr_dB)
    # if the CQI that guarantees 10% BLER is lower than the CQI from the Shannon based CQI
    if t_BLER_cqi_i < cqi_i:
      t_BLER_cqi_Qm, t_BLER_cqi_R, t_BLER_cqi_SE = _CQI_TABLE_1[t_BLER_cqi_i, 1:][0]
      df_mcs = df_mcs[df_mcs['SE'] <= t_BLER_cqi_SE]
  # Get the row with the maximum SE
  df_mcs = df_mcs[df_mcs['SE'] == df_mcs['SE'].max()]
  mcs_index = df_mcs.index[0]
  mcs_Qm, mcs_R, mcs_SE = df_mcs.iloc[0]

  if mcs_SE > se_Shannon:
    print('Oh-oh! Something has gone wrong.')
    print(color_text.Red(f'  The MCS SE is greater than the Shannon SE:\n\tMCS SE: {mcs_SE}\n\tShannon SE: {se_Shannon}'))
    return None, None, None, None
  return mcs_index, mcs_Qm, mcs_R, mcs_SE

def SE_to_MCS_Qm_R_SE_vect(subband_CQI_Qm_R_SE:np.array, dbg=False): # OOP implementation for AIMM. Deprecate? (KS 2024-11-15)
  """A vectorized version that can handle subbands."""
  MCS_Table_1=MCS_TABLE_1_FOR_PDSCH_ARRAY
  # Make a copy of the subband_CQI_Qm_R_SE array
  CQI_array=subband_CQI_Qm_R_SE.copy()
  cqi_array_qm=CQI_array[:,0,1]
  cqi_array_se=CQI_array[:,0,3]
  mcs_qm=MCS_Table_1[:,1]
  mcs_se=MCS_Table_1[:,3]
  mask=(mcs_qm[None, :] == cqi_array_qm[:, None]) & (mcs_se[None, :] <= cqi_array_se[:, None])

  # Get indices of the last valid match along each row in new_array (a bit optimisitc really)
  # Reverse the mask along the columns, find the first valid match (which is the last valid match in the original order)
  mcs_indices = np.argmax(mask[:, ::-1], axis=1)
  mcs_indices = mask.shape[1] - 1 - mcs_indices  # Convert reversed indices to original indices
  valid_mcs_found = np.any(mask, axis=1)  # Check if any valid match exists per row

  # Create empty mcs_array
  mcs_array=np.full_like(CQI_array, np.nan)
  # For the rows with valid MCS entries, assign the corresponding MCS row
  mcs_array[valid_mcs_found, 0,:]=MCS_Table_1[mcs_indices[valid_mcs_found]]

  # Print the MCS array.
  if dbg: print(f'MCS array:\n\t{mcs_array}')
  return mcs_array

@dataclass
class Radio_state:
  NofSlotsPerRadioFrame: int=20
  NofRadioFramePerSec: int  =100
  nCC: int                  =1                # Number of of component carriers
  mu: int                   =0                # Numerology
  NRB_sc: int               =12               # Number of subcarriers per PRB
  Nsh_symb: int             =(14*(2**mu)-1)   # Number of symbols of the PDSCH allocation within the slot
  NPRB_oh: float            =0.0              # Overhead of the PRB (should be from set {0,6,12,18}). Default=0
  nPRB: int                 =273              # Number of PRBs allocated to the UE
  Qm: int                   =8                # Modulation order
  v: int                    =4                # Number of Layers (i.e. spatial streams. Based on n_antennas at Tx and Rx)
  R: float                  =0.948            # Code rate
  MCS: int                  =20               # Modulation and Coding Scheme (MCS) index
  operating_mode: str       ='TDD'            # 'TDD' or 'FDD' based on the centre frequency of the carrier (fc_GHz). 3.5 GHz is TDD.
  dl_ul_ratio: float        =10/14            # Downlink to uplink ratio


# FUNCTIONS by Keith Briggs

def SINR_to_CQI(sinr_dB):
  return np.searchsorted(SINR90PC,sinr_dB)-1 # vectorized

def CQI_to_efficiency_QPSK(cqi):
  # non-vectorized (TODO)
  if not 0<=cqi<=15: return float('nan')
  return _CQI_TABLE_1[cqi,-1]

def RSRP_report(rsrp_dBm):
  '''
    Convert RSRP report from dBm to standard range.

    Parameters
    ----------
    rsrp_dBm : float
        RSRP report in dBm 

    Returns
    -------
    int
        RSRP report in standard range.
  '''
  if rsrp_dBm==float('inf'): return 127
  if rsrp_dBm<-156.0: return 0
  if rsrp_dBm>=-31.0: return 126
  return int(rsrp_dBm+156.0)

def max_5G_throughput_64QAM(radio_state):
  # https://www.sharetechnote.com/html/5G/5G_MCS_TBS_CodeRate.html
  # converted from octave/matlab Keith Briggs 2020-10-09
  Qm,R,Spectral_efficiency=MCS_TABLE_1_FOR_PDSCH[radio_state.MCS]
  R/=1024.0 # MCS_to_Qm_table has 1024*R
  NPRB_DMRS=_DMRS_RE('type1','A',1,0)
  NREprime=radio_state.NRB_sc*radio_state.Nsh_symb-NPRB_DMRS-radio_state.NPRB_oh
  NREbar=min(156,NREprime)
  NRE=NREbar*radio_state.nPRB
  Ninfo=NRE*R*Qm*radio_state.v
  if Ninfo>3824:
    n=int(log2(Ninfo-24))-5
    Ninfo_prime=2**n*round((Ninfo-24)/(2**n))
    if R>0.25:
      C=ceil((Ninfo_prime+24)/8424) if Ninfo_prime>=8424 else 1.0
    else: # R<=1/4
      C=ceil((Ninfo_prime+24)/3816)
    TBS_bits=8*C*ceil((Ninfo_prime+24)/(8*C))-24
  else: # Ninfo<=3824
    Ninfo=max(24,2**n*int(Ninfo/2**n))
    print('Ninfo<=3824 not yet implemented - need 38.214 Table 5.1.3.2-2')
    sys.exit(1)
  TP_bps=TBS_bits*radio_state.NofSlotsPerRadioFrame*radio_state.NofRadioFramePerSec
  return TP_bps/1024/1024

def _DMRS_RE(typ,mapping,length,addPos):
  # https://www.sharetechnote.com/html/5G/5G_MCS_TBS_CodeRate.html#PDSCH_TBS
  # converted from octave/matlab Keith Briggs 2020-10-09
  if typ=='type1':
    DMRSType='type1'
    if mapping=='A':  
      PDSCH_MappingType='A'
      if   addPos==0: dmrsRE=  6*length
      elif addPos==1: dmrsRE=2*6*length
      elif addPos==2: dmrsRE=3*6*length
      elif addPos==3: dmrsRE=4*6*length
      AdditionalPos=addPos
    elif mapping=='B': dmrsRE=6*length
  else:
    DMRSType='type2'
    if mapping=='B':
      PDSCH_MappingType='A'
      if   addPos==0: dmrsRE=  4*length
      elif addPos==1: dmrsRE=2*4*length
      elif addPos==2: dmrsRE=3*4*length
      elif addPos==3: dmrsRE=4*4*length
      AdditionalPos=addPos
    elif mapping=='A': dmrsRE=4*length; # FIXME is 'A' right here?
  return dmrsRE

# plot functions - only for testing...

def plot_SINR_to_CQI(fn='../img/plot_SINR_to_CQI'):
  n,bot,top=1000,-10.0,35.0
  x=np.linspace(bot,top,n)
  y=[SINR_to_CQI(x[i]) for i in range(n)]
  # write table to tab-separated file...
  f=open('SINR_to_CQI_table.tsv','w')
  for xy in zip(x,y): f.write('%.3f\t%.3f\n'%xy)
  f.close()
  fig=plt.figure()
  ax=fig.add_subplot(1,1,1)
  ax.set_xlim(bot,top)
  ax.set_ylim(0,15)
  ax.grid(linewidth=1,color='gray',alpha=0.25)
  ax.scatter(x,y,marker='.',s=1,label='exact SINR to CQI mapping',color='red')
  ax.plot([-5,29],[0,15],':',color='blue',alpha=0.7,label='linear approximation')
  ax.set_xlabel('SINR (dB)')
  ax.set_ylabel('CQI')
  ax.legend(loc='lower right')
  fig.tight_layout()
  fig_timestamp(fig,author='Keith Briggs',rotation=90)
  fig.savefig('%s.png'%fn)
  fig.savefig('%s.pdf'%fn)
  print('eog %s.png &'%fn)
  print('evince %s.pdf &'%fn)

def plot_CQI_to_efficiency_QPSK(fn='../img/plot_CQI_to_efficiency_QPSK'):
  bot,top=0,15
  x=range(bot,1+top)
  y=[CQI_to_efficiency_QPSK(xi) for xi in x]
  fig=plt.figure()
  ax=fig.add_subplot(1,1,1)
  ax.set_xlim(1+bot,top)
  ax.set_ylim(0,6)
  ax.grid(linewidth=1,color='gray',alpha=0.25)
  ax.scatter(x,y,marker='o',s=2,label='CQI to efficiency (QPSK)',color='red')
  ax.plot(x,y,':',color='gray',alpha=0.5)
  ax.set_xlabel('CQI')
  ax.set_ylabel('spectral efficiency')
  ax.legend(loc='lower right')
  fig.tight_layout()
  fig_timestamp(fig,author='Keith Briggs',rotation=90)
  fig.savefig('%s.png'%fn)
  fig.savefig('%s.pdf'%fn)
  print('eog %s.png &'%fn)
  print('evince %s.pdf &'%fn)

# better 2021-03-08 (cannot easily vectorize)...
@lru_cache(maxsize=None)
def CQI_to_64QAM_efficiency(cqi):
  CQI_to_MCS=max(0,min(28,int(28*cqi/15.0)))
  return CQI_to_MCS, MCS_TABLE_1_FOR_PDSCH[CQI_to_MCS][-1]

def plot_CQI_to_efficiency(fn='../img/plot_CQI_to_efficiency'):
  # TODO 256QAM
  bot,top=0,15
  cqi=range(bot,1+top)
  y=[CQI_to_64QAM_efficiency(x) for x in cqi]
  fig=plt.figure()
  ax=fig.add_subplot(1,1,1)
  ax.set_xlim(bot,top)
  ax.set_ylim(ymin=0,ymax=6)
  ax.grid(linewidth=0.5,color='gray',alpha=0.25)
  ax.plot(cqi,y,':',color='gray',ms=0.5,alpha=0.7)
  ax.scatter(cqi,y,marker='o',s=9,label='efficiency (64 QAM)',color='red')
  ax.set_xlabel('CQI')
  ax.set_ylabel('spectral efficiency')
  ax.legend(loc='lower right')
  fig.tight_layout()
  fig_timestamp(fig,author='Keith Briggs',rotation=90)
  fig.savefig('%s.png'%fn)
  fig.savefig('%s.pdf'%fn)
  print('eog %s.png &'%fn)
  print('evince %s.pdf &'%fn)

if __name__=='__main__':
  plot_CQI_to_efficiency_QPSK()
  plot_SINR_to_CQI()
  pass
