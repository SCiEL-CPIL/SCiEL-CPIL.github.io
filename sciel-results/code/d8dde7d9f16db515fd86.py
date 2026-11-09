"""
Symbolic regression for stress-strain of an aluminium rod.
Uses a piecewise-linear approximation with ramp functions for efficient BFGS fitting.
The 10 parameters are fitted by the evaluator using BFGS.
"""
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Model structure:
    - Piecewise-linear approximation using clipped linear terms (ramps)
    - Ramps allow modeling of yield points and hardening regimes
    - Temperature effects modify slopes and intercepts
    
    Parameters:
    p0: intercept
    p1: elastic slope (Young's modulus)
    p2: yield strain location (kink 1)
    p3: hardening strain location (kink 2)
    p4: plastic hardening slope
    p5: temperature scaling for elastic slope
    p6: temperature scaling for yield stress
    p7: temperature scaling for hardening slope
    p8: additional temperature interaction term
    p9: high-temperature offset
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Center temperature for better conditioning
    temp_c = temp - 0.5
    
    # Temperature-dependent kink locations (yield and hardening transition)
    kink1 = p2 + p3 * temp_c
    kink2 = p4 + p5 * temp_c
    
    # Temperature-dependent slopes
    E_T = p1 * (1.0 + p6 * temp_c)
    H1_T = p7 * (1.0 + p8 * temp_c)
    
    # Three ramp functions for piecewise-linear approximation
    ramp1 = np.maximum(strain - kink1, 0.0)
    ramp2 = np.maximum(strain - kink2, 0.0)
    
    # Base stress: elastic + two hardening regimes
    stress = p0 + E_T * strain + H1_T * ramp1 + p9 * ramp2
    
    return stress
