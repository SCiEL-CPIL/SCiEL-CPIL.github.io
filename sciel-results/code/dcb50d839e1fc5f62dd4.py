"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with time-dependent driving force.
    
    Structure: a = -k(x)*x - c(v)*v + F(t)
    where k(x) and c(v) are smooth bounded functions using tanh.
    
    This parametric limit-cycle oscillator structure enforces physical stability
    and nonlinearity, constraining the solution space to stable dynamical regimes.
    
    Features:
    0: bias
    1: linear restoring (k0 * x)
    2: nonlinear restoring (k1 * x * tanh(x) * x) -> effectively x*tanh(x)
    3: linear damping (c0 * v)
    4: nonlinear damping (c1 * v * tanh(v) * v) -> effectively v*tanh(v)
    5: sin(t) drive
    6: cos(t) drive
    7: x*v coupling
    8: x^2 term
    9: parametric coupling sin(t)*x
    """
    # Normalize inputs to help BFGS convergence
    # t: [30, 50] -> center 40, scale 10
    # x: [-0.2056, 0.1749] -> center -0.015, scale 0.2
    # v: [-0.3391, 0.3624] -> center 0.01, scale 0.35
    
    t_n = (t - 40.0) / 10.0
    x_n = (x + 0.015) / 0.2
    v_n = (v - 0.01) / 0.35
    
    # Bounded nonlinearities using tanh
    tanh_x = np.tanh(x_n)
    tanh_v = np.tanh(v_n)
    
    # Nonlinear restoring: x * tanh(x) gives smooth bounded nonlinearity
    nonlinear_x = x_n * tanh_x
    
    # Nonlinear damping: v * tanh(v) gives smooth bounded nonlinearity
    nonlinear_v = v_n * tanh_v
    
    # Driving force components
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Coupling terms
    xv = x_n * v_n
    x2 = x_n * x_n
    sin_t_x = sin_t * x_n
    
    # Combine features with params
    # p0: bias
    # p1: linear restoring coefficient
    # p2: nonlinear restoring coefficient (Duffing-like but bounded)
    # p3: linear damping coefficient
    # p4: nonlinear damping coefficient (bounded)
    # p5: drive sin amplitude
    # p6: drive cos amplitude
    # p7: coupling xv
    # p8: quadratic x
    # p9: parametric coupling sin(t)*x
    
    return (params[0] + 
            params[1] * x_n + 
            params[2] * nonlinear_x + 
            params[3] * v_n + 
            params[4] * nonlinear_v + 
            params[5] * sin_t + 
            params[6] * cos_t + 
            params[7] * xv + 
            params[8] * x2 + 
            params[9] * sin_t_x)
# EVOLVE-BLOCK-END
