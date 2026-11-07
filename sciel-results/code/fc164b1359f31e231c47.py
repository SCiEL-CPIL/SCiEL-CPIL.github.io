"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod using Ramberg-Osgood formulation.
    
    sigma = E(T) * eps + K(T) * |eps|^n * sign(eps)
    
    Where:
    - E(T) = p0 * (1 + p1 * (T - Tref))  : temperature-dependent Young's modulus
    - K(T) = p2 * (1 + p3 * (T - Tref))  : temperature-dependent hardening coefficient
    - n = p4                               : hardening exponent
    - p5, p6, p7, p8, p9: additional basis functions for flexibility
    
    The model is designed to be smooth and physically interpretable.
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    s = strain
    t = temp
    p = params
    
    # Reference temperature (middle of training range)
    Tref = 0.5
    
    # Temperature-dependent parameters
    # E(T) = p0 * (1 + p1 * (T - Tref))
    E = p[0] * (1.0 + p[1] * (t - Tref))
    
    # K(T) = p2 * (1 + p3 * (T - Tref))
    K = p[2] * (1.0 + p[3] * (t - Tref))
    
    # Hardening exponent
    n = p[4]
    
    # Clamp n to avoid numerical issues (n should be > 0)
    n_safe = np.maximum(n, 0.01)
    
    # Ramberg-Osgood: sigma = E * eps + K * |eps|^n * sign(eps)
    abs_s = np.abs(s)
    sign_s = np.sign(s)
    
    # Handle zero strain carefully
    eps_min = 1e-10
    abs_s_safe = np.maximum(abs_s, eps_min)
    
    # Nonlinear term: K * |eps|^n
    nonlinear = K * np.power(abs_s_safe, n_safe)
    
    # Full stress
    sigma = E * s + nonlinear * sign_s
    
    # Add additional basis functions for more flexibility
    # These capture higher-order effects and interactions
    # p5: quadratic strain term
    # p6: strain-temperature interaction
    # p7: cubic strain term
    # p8: temperature squared term
    # p9: linear temperature offset
    
    sigma = sigma + p[5] * s * s
    sigma = sigma + p[6] * s * (t - Tref)
    sigma = sigma + p[7] * s * s * s
    sigma = sigma + p[8] * (t - Tref) ** 2
    sigma = sigma + p[9] * (t - Tref)
    
    # Check deadline
    if time.time() > deadline:
        return sigma
    
    return sigma
# EVOLVE-BLOCK-END
