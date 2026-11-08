"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator, from position and velocity.
    Uses rational parameterization: params define shape of state-dependent
    stiffness and damping, not just linear scaling of fixed basis functions.
    """
    p = params
    
    # Scale inputs to O(1) in training range
    x_s = x * 1.5
    v_s = v * 4.0
    
    # Core quadratic/cubic terms
    x2 = x_s * x_s
    x3 = x2 * x_s
    v2 = v_s * v_s
    v3 = v2 * v_s
    
    # Cross terms
    xv = x_s * v_s
    x2v = x2 * v_s
    
    # Smooth saturating functions
    tanh_x = np.tanh(x_s)
    tanh_v = np.tanh(v_s)
    
    # Rational form: state-dependent stiffness k(x) = (p0 + p1*x2 + p2*x4) / (1 + p3*x2)
    # This lets params control the *shape* of the restoring force
    x4 = x2 * x2
    stiffness_num = p[0] + p[1] * x2 + p[2] * x4
    stiffness_den = 1.0 + p[3] * x2
    # Safe division
    k_eff = stiffness_num / np.maximum(stiffness_den, 1e-8)
    restoring = k_eff * x_s  # odd in x
    
    # Rational form: state-dependent damping c(v) = (p4 + p5*v2) / (1 + p6*v2)
    damping_num = p[4] + p[5] * v2
    damping_den = 1.0 + p[6] * v2
    c_eff = damping_num / np.maximum(damping_den, 1e-8)
    damping = c_eff * v_s  # odd in v
    
    # Coupling terms with rational modulation
    # p7: linear coupling strength
    # p8: modulates coupling by position (velocity-dependent stiffness)
    coupling = p[7] * xv + p[8] * x2v
    
    # p9: bias / offset
    bias = p[9]
    
    y = restoring + damping + coupling + bias
    
    return y
# EVOLVE-BLOCK-END
