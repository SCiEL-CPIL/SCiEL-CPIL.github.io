"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model:
    mu = mu_max * Monod(s) * TempOpt(temp) * PHOpt(pH) * DensityInhib(b)
    
    Uses smooth, differentiable functions suitable for BFGS optimization.
    """
    # Unpack 10 parameters with safe defaults
    mu_max = params[0]      # Maximum growth rate
    Ks = params[1]          # Substrate half-saturation constant
    T_opt = params[2]       # Optimal temperature
    T_width = np.abs(params[3]) + 1e-6  # Temperature width (keep positive)
    pH_opt = params[4]      # Optimal pH
    pH_width = np.abs(params[5]) + 1e-6 # pH width (keep positive)
    b_max = np.abs(params[6]) + 1e-6   # Max density before inhibition kicks in
    inhib_strength = params[7]         # Strength of density inhibition
    shift = params[8]                  # Vertical shift (baseline)
    scale = np.abs(params[9]) + 1e-6   # Overall scale factor
    
    # Monod kinetics for substrate: s / (Ks + s)
    Ks_safe = np.abs(Ks) + 1e-6
    monod = s / (Ks_safe + s)
    
    # Temperature optimum: Gaussian-like decay from optimal temp
    # Using exp(-((temp - T_opt) / T_width)^2) for smoothness
    temp_factor = np.exp(-((temp - T_opt) / T_width) ** 2)
    
    # pH optimum: Gaussian-like decay from optimal pH
    ph_factor = np.exp(-((pH - pH_opt) / pH_width) ** 2)
    
    # Density inhibition: 1 / (1 + (b / b_max)^n)
    # Use n=2 for smooth logistic-like inhibition
    density_factor = 1.0 / (1.0 + (b / b_max) ** 2)
    
    # Combine multiplicatively
    mu = mu_max * monod * temp_factor * ph_factor * density_factor
    
    # Apply scale and shift
    mu = mu * scale + shift
    
    # Ensure output is finite
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return mu
# EVOLVE-BLOCK-END
