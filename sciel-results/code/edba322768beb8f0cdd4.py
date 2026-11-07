"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rich basis symbolic regression:
    - Polynomial up to cubic
    - Sin/cos with learnable frequencies (2 pairs)
    - Exp with learnable rate
    - Tanh with learnable rate
    
    params layout (10 elements):
      params[0]:  a0 (constant)
      params[1]:  a1 (x coeff)
      params[2]:  a2 (x^2 coeff)
      params[3]:  a3 (x^3 coeff)
      params[4]:  b1 * sin(f1 * x)  -> b1
      params[5]:  f1 (freq for sin)
      params[6]:  b2 * cos(f2 * x)  -> b2
      params[7]:  f2 (freq for cos)
      params[8]:  c1 * exp(r1 * x)  -> c1
      params[9]:  r1 (rate for exp)
    """
    a0 = params[0]
    a1 = params[1]
    a2 = params[2]
    a3 = params[3]
    b1 = params[4]
    f1 = params[5]
    b2 = params[6]
    f2 = params[7]
    c1 = params[8]
    r1 = params[9]
    
    poly = a0 + a1 * x + a2 * x**2 + a3 * x**3
    trig = b1 * np.sin(f1 * x) + b2 * np.cos(f2 * x)
    exp_term = c1 * np.exp(np.clip(r1 * x, -10, 10))  # clip to avoid overflow
    
    return poly + trig + exp_term
# EVOLVE-BLOCK-END
