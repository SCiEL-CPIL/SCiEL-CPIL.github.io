# EVOLVE-BLOCK-START
import numpy as np


def _stable_logistic(z):
    """Compute 1 / (1 + exp(z)) stably."""
    # Clip z to prevent overflow in exp
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(z))


def _gen_logistic(x, x0, k, q):
    """Generalized logistic function: 1 / (1 + exp(-k(x-x0)))^q"""
    z = -k * (x - x0)
    base = _stable_logistic(z)
    # q should be positive to keep function bounded and peak-like if k is signed appropriately
    # If q is negative, it might blow up. Let's force q > 0.
    q_pos = np.abs(q) + 1e-6
    return base ** q_pos


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate using Haldane/Logistic structural model."""
    # Unpack parameters
    # 1: Ks (substrate half-saturation)
    # 2,3,4: T_opt, k_T, q_T (Temperature logistic)
    # 5,6,7: pH_opt, k_pH, q_pH (pH logistic)
    # 8: K_b (density half-saturation)
    # 9: mu_max
    # 10: offset
    
    Ks = np.abs(params[0]) + 1e-6
    T_opt = params[1]
    k_T = params[2]
    q_T = params[3]
    
    pH_opt = params[4]
    k_pH = params[5]
    q_pH = params[6]
    
    K_b = np.abs(params[7]) + 1e-6
    mu_max = params[8]
    offset = params[9]

    # Substrate: Monod
    S_term = s / (Ks + s)
    
    # Temperature: Generalized Logistic
    T_term = _gen_logistic(temp, T_opt, k_T, q_T)
    
    # pH: Generalized Logistic
    pH_term = _gen_logistic(pH, pH_opt, k_pH, q_pH)
    
    # Density: Monod
    b_term = b / (K_b + b)
    
    # Combine
    growth = mu_max * S_term * T_term * pH_term * b_term
    
    return growth + offset
# EVOLVE-BLOCK-END
