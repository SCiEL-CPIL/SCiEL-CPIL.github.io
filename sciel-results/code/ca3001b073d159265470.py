"""Symbolic regression: E. coli growth rate model with biologically-grounded structure."""
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model combining:
    - Monod kinetics for substrate saturation
    - Gaussian temperature response centered near physiological optimum
    - Gaussian pH response centered near neutral
    - Density-dependent inhibition (logistic-like)
    
    Parameters:
    params[0]: max growth rate (mu_max)
    params[1]: half-saturation constant for substrate (Ks)
    params[2]: temperature optimum (T_opt)
    params[3]: temperature width (sigma_T)
    params[4]: pH optimum (pH_opt)
    params[5]: pH width (sigma_pH)
    params[6]: carrying capacity / max density (K)
    params[7]: density inhibition coefficient
    params[8]: baseline offset
    params[9]: additional scaling factor
    """
    mu_max = params[0]
    Ks = params[1]
    T_opt = params[2]
    sigma_T = params[3]
    pH_opt = params[4]
    sigma_pH = params[5]
    K = params[6]
    alpha_b = params[7]
    offset = params[8]
    scale = params[9]
    
    # Safety: avoid division by zero and negative variances
    Ks_safe = np.maximum(np.abs(Ks), 1e-6)
    sigma_T_safe = np.maximum(np.abs(sigma_T), 1e-6)
    sigma_pH_safe = np.maximum(np.abs(sigma_pH), 1e-6)
    K_safe = np.maximum(np.abs(K), 1e-6)
    
    # Monod substrate saturation: s / (Ks + s)
    monod = s / (Ks_safe + s)
    
    # Gaussian temperature response
    temp_response = np.exp(-((temp - T_opt) ** 2) / (2 * sigma_T_safe ** 2))
    
    # Gaussian pH response
    pH_response = np.exp(-((pH - pH_opt) ** 2) / (2 * sigma_pH_safe ** 2))
    
    # Density-dependent inhibition: 1 - (b/K)^2 (quadratic inhibition)
    density_inhibition = 1.0 - (b / K_safe) ** 2
    # Clip to avoid negative growth from excessive inhibition
    density_inhibition = np.maximum(density_inhibition, 0.0)
    
    # Combine multiplicatively
    growth = mu_max * monod * temp_response * pH_response * density_inhibition * scale + offset
    
    return growth
