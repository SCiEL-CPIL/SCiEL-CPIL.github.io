import numpy as np
import time
from scipy.optimize import minimize


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Parametric limit-cycle oscillator:
    
    a = -k(x)*x - c(v)*v + F(t) + coupling + offset
    
    where:
    k(x) = p0 + p1*tanh(p2*x)      (position-dependent stiffness, bounded)
    c(v) = p3 + p4*tanh(p5*v)      (velocity-dependent damping, bounded)
    F(t) = p6*cos(t) + p7*sin(t)   (periodic driving)
    coupling = p8*x*v              (parametric coupling)
    offset = p9
    
    Inputs are normalized to unit scale for better BFGS conditioning.
    All 10 parameters are fitted via BFGS starting from all ones.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to roughly unit scale for better conditioning
    # x range: [-0.2056, 0.1749], half-range ~0.19
    # v range: [-0.3391, 0.3624], half-range ~0.35
    x_norm = x / 0.19
    v_norm = v / 0.35
    
    # Bounded stiffness: k(x) = p0 + p1*tanh(p2*x_norm)
    k_x = p0 + p1 * np.tanh(p2 * x_norm)
    
    # Bounded damping: c(v) = p3 + p4*tanh(p5*v_norm)
    c_v = p3 + p4 * np.tanh(p5 * v_norm)
    
    # Periodic driving force
    F_t = p6 * np.cos(t) + p7 * np.sin(t)
    
    # Parametric coupling
    coupling = p8 * x * v
    
    # Limit-cycle oscillator: a = -k(x)*x - c(v)*v + F(t) + coupling + offset
    return (-k_x * x - c_v * v + F_t + coupling + p9)
