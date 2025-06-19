"""
dB_conversions_01.py
-----------------
This module provides utility functions for converting between decibel (dB), power ratios, and various units of frequency and power commonly used in signal processing and telecommunications.

Authors: Keith Briggs & Kishan Sthankiya.
"""
from numpy import log10,power,where,inf, asarray, nan, isfinite
import numpy as np

def watts_to_dB(x):  
    """
    Convert power values from watts to decibels (dB).

    Parameters
    ----------
    x : array_like or float
        Input power value(s) in watts.

    Returns
    -------
    dB : ndarray or float
        Power value(s) in decibels (dB). Returns -inf where input is zero.

    Notes
    -----
    The conversion is defined as 10 * log10(x). For zero input, returns -inf.
    """
    return where(x!=0, 10.0*log10(x,where=(x!=0)), -inf)

def dBm_to_watts(x): 
    """
    Convert power from decibel-milliwatts (dBm) to watts.

    Parameters
    ----------
    x : float or array-like
        Power level(s) in dBm.

    Returns
    -------
    float or ndarray
        Power level(s) in watts.

    Notes
    -----
    The conversion is performed using the formula:
        P(W) = 10 ** ((P(dBm) - 30) / 10)
    """
    return power(10.0,(x-30.0)/10.0)

def dB_to_watts(x):  
    """
    Convert decibel (dB) value to power in watts.

    Parameters
    ----------
    x : float
        Power level in decibels (dB).

    Returns
    -------
    float
        Power in watts corresponding to the input dB value.
    """
    return power(10.0,x/10.0)

def dB_to_ratio(x): 
    """
    Convert a value from decibels (dB) to linear ratio.

    Parameters
    ----------
    x : float or array-like
        Value(s) in decibels (dB) to be converted.

    Returns
    -------
    float or ndarray
        Linear ratio corresponding to the input dB value(s).

    Notes
    -----
    This function assumes a power ratio conversion: ratio = 10^(x/10).
    """
    return power(10.0,x/10.0)

def milliwatts_to_watts(x): 
    """
    Convert power from milliwatts to watts.

    Parameters
    ----------
    x : float or array-like
        Power value(s) in milliwatts.

    Returns
    -------
    float or array-like
        Power value(s) in watts.
    """
    return 1e-3 * x

def watts_to_milliwatts(x): 
    """
    Convert power from watts to milliwatts.

    Parameters
    ----------
    x : float or array-like
        Power value(s) in watts.

    Returns
    -------
    float or array-like
        Power value(s) in milliwatts.
    """
    return 1e3 * x

def MHz_to_Hz(x): 
    """
    Convert a value from megahertz (MHz) to hertz (Hz).

    Parameters
    ----------
    x : float or array-like
        Value(s) in megahertz.

    Returns
    -------
    float or ndarray
        Equivalent value(s) in hertz.
    """
    return 1e6 * x

def Hz_to_MHz(x): 
    """
    Convert frequency from Hertz (Hz) to Megahertz (MHz).

    Parameters
    ----------
    x : float or array-like
        Frequency value(s) in Hertz.

    Returns
    -------
    float or array-like
        Frequency value(s) in Megahertz.
    """
    return 1e-6 * x

def GHz_to_Hz(x): 
    """
    Convert frequency from gigahertz (GHz) to hertz (Hz).

    Parameters
    ----------
    x : float or array-like
        Frequency value(s) in gigahertz.

    Returns
    -------
    float or ndarray
        Frequency value(s) in hertz.
    """
    return 1e9 * x

def Hz_to_GHz(x): 
    """
    Convert frequency from Hertz (Hz) to Gigahertz (GHz).

    Parameters
    ----------
    x : float or array-like
        Frequency value(s) in Hertz.

    Returns
    -------
    float or array-like
        Frequency value(s) in Gigahertz.
    """
    return 1e-9 * x

def kHz_to_Hz(x): 
    """
    Convert frequency from kilohertz (kHz) to hertz (Hz).

    Parameters
    ----------
    x : float or array-like
        Frequency value(s) in kilohertz.

    Returns
    -------
    float or ndarray
        Frequency value(s) converted to hertz.
    """
    return 1e3 * x

def Hz_to_kHz(x):
    """
    Convert frequency from Hertz (Hz) to kilohertz (kHz).

    Parameters
    ----------
    x : float or array-like
        Frequency value(s) in Hertz.

    Returns
    -------
    float or array-like
        Frequency value(s) in kilohertz.
    """
    return 1e-3 * x

def ratio_to_dB(x):
    """
    Convert a ratio to decibels (dB).

    Parameters
    ----------
    x : array_like
        Input value(s) representing the ratio(s) to be converted to decibels.
        Must be positive; non-positive values will be replaced with NaN.

    Returns
    -------
    dB : ndarray or scalar
        The corresponding value(s) in decibels (dB). For non-finite input values,
        returns -inf.

    Notes
    -----
    The conversion is performed using the formula: dB = 10 * log10(x).
    Non-positive values in `x` are replaced with NaN before conversion.
    """
    x = asarray(x)  
    x = where(x > 0, x, nan)  # Replace non-positive values with NaN
    return where(isfinite(x), 10.0 * log10(x), -inf)

def watts_to_dBm(x):
    """
    Convert power from watts to decibel-milliwatts (dBm).

    Parameters
    ----------
    x : array_like or float
        Input power value(s) in watts.

    Returns
    -------
    dBm : ndarray or float
        Power value(s) in dBm. Non-positive inputs are replaced with NaN and return -inf.

    Notes
    -----
    - For non-positive input values, the function returns -inf.
    - Uses the formula: dBm = 30 + 10 * log10(x).
    """
    x = asarray(x)
    x = where(x > 0, x, nan)  # Replace non-positive values with NaN
    return where(isfinite(x), 30.0 + 10.0 * log10(x), -inf)

def test_functions():
    watts = np.array([0, 1e-3, 1, 10])
    dBm = np.array([-30, 0, 10, 30])
    dB = np.array([-10, 0, 10, 20])
    ratios = np.array([0, 0.1, 1, 10])
    mW = np.array([0, 1, 1000])
    MHz = np.array([0, 1, 100])
    GHz = np.array([0, 1, 2.5])
    kHz = np.array([0, 1, 1000])
    Hz = np.array([0, 1e3, 1e6, 1e9])

    tests = [
        ("watts_to_dB", watts_to_dB, watts),
        ("dBm_to_watts", dBm_to_watts, dBm),
        ("dB_to_watts", dB_to_watts, dB),
        ("dB_to_ratio", dB_to_ratio, dB),
        ("milliwatts_to_watts", milliwatts_to_watts, mW),
        ("watts_to_milliwatts", watts_to_milliwatts, watts),
        ("MHz_to_Hz", MHz_to_Hz, MHz),
        ("Hz_to_MHz", Hz_to_MHz, Hz),
        ("GHz_to_Hz", GHz_to_Hz, GHz),
        ("Hz_to_GHz", Hz_to_GHz, Hz),
        ("kHz_to_Hz", kHz_to_Hz, kHz),
        ("Hz_to_kHz", Hz_to_kHz, Hz),
        ("ratio_to_dB", ratio_to_dB, ratios),
        ("watts_to_dBm", watts_to_dBm, watts),
    ]

    for name, func, vals in tests:
        print(f"{name}: {func(vals)}")

if __name__ == "__main__":
    test_functions()