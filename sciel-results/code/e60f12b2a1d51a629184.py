"""Nonlinear oscillator symbolic regression with BFGS-friendly parameterization."""
import numpy as np
import time

# Constants for input scaling to O(1)
_T_MIN, _T_MAX = 30.0, 50.0
_X_MIN, _X_MAX = -0.2056, 0.1749
_V_MIN, _V_MAX = -0.3391, 0.3624

_T_SCALE = (_T_MAX - _T_MIN) / 2.0
_T_OFFSET = (_T_MAX + _T_MIN) / 2.0
_X_SCALE = (_X_MAX - _X_MIN) / 2.0
_X_OFFSET = (_X_MAX + _X_MIN) / 2.0
_V_SCALE = (_V_MAX - _V_MIN) / 2.0
_V_OFFSET = (_V_MAX + _V_MIN) / 2.0


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear damped oscillator with time-dependent driving.
    
    Model: a = c0 + c1*x + c2*x^2 + c3*x^3 + c4*v + c5*v^2 + c6*x*v + c7*t + c8*t*x + c9*t*v
    
    Parameters are transformed for BFGS stability:
    - Linear terms (c0, c1, c4, c7) use direct mapping
    - Nonlinear terms use tanh scaling to keep magnitudes reasonable
    """
    p = params
    
    # Scale inputs to approximately [-1, 1]
    ts = (t - _T_OFFSET) / _T_SCALE
    xs = (x - _X_OFFSET) / _X_SCALE
    vs = (v - _V_OFFSET) / _V_SCALE
    
    # Extract parameters with appropriate scaling for BFGS convergence
    # p[0]: constant offset (linear, direct)
    c0 = p[0]
    # p[1]: linear x coefficient (scaled by range to keep gradient balanced)
    c1 = p[1] * _X_SCALE
    # p[2]: x^2 coefficient (nonlinear, use tanh for bounded influence)
    c2 = p[2] * _X_SCALE * _X_SCALE
    # p[3]: x^3 coefficient (nonlinear)
    c3 = p[3] * _X_SCALE * _X_SCALE * _X_SCALE
    # p[4]: linear v coefficient (damping, direct)
    c4 = p[4] * _V_SCALE
    # p[5]: v^2 coefficient (nonlinear damping)
    c5 = p[5] * _V_SCALE * _V_SCALE
    # p[6]: x*v interaction
    c6 = p[6] * _X_SCALE * _V_SCALE
    # p[7]: linear t coefficient (driving)
    c7 = p[7] * _T_SCALE
    # p[8]: t*x interaction
    c8 = p[8] * _T_SCALE * _X_SCALE
    # p[9]: t*v interaction
    c9 = p[9] * _T_SCALE * _V_SCALE
    
    # Compute the model
    result = (c0 
              + c1 * xs 
              + c2 * xs**2 
              + c3 * xs**3 
              + c4 * vs 
              + c5 * vs**2 
              + c6 * xs * vs 
              + c7 * ts 
              + c8 * ts * xs 
              + c9 * ts * vs)
    
    return result
