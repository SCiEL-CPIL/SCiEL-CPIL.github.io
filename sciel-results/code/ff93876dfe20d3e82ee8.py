"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using raw inputs with multiplicative structure.
    
    Combines:
    - Monod kinetics for substrate (s)
    - Gaussian response for temperature (temp)
    - Gaussian response for pH
    - Linear response for population density (b)
    - Interaction terms for curvature
    
    Uses raw inputs (no normalization) with parameters directly on the data scale.
    
    Parameters (10 used):
    p0: log(C) -> Max growth rate scale
    p1: K -> Substrate half-saturation constant (raw scale)
    p2: T_center -> Optimal temperature (raw scale)
    p3: sigma_T -> Temperature response width (raw scale)
    p4: pH_center -> Optimal pH (raw scale)
    p5: sigma_pH -> pH response width (raw scale)
    p6: gamma -> Density coefficient
    p7: beta_bs -> b*s interaction coefficient
    p8: beta_TpH -> temp*pH interaction coefficient
    p9: B -> Bias
    """
    # Unpack params with safe transforms
    c_log = np.clip(params[0], -15, 15)
    K = np.exp(np.clip(params[1], -10, 10)) + 1e-6
    t_center = params[2]
    sig_t = np.exp(np.clip(params[3], -10, 10)) + 1e-6
    ph_center = params[4]
    sig_ph = np.exp(np.clip(params[5], -10, 10)) + 1e-6
    gamma = params[6]
    beta_bs = params[7]
    beta_TpH = params[8]
    bias = params[9]

    C = np.exp(c_log)

    # Monod term on raw scale
    monod = s / (K + s + 1e-9)

    # Gaussian terms on raw scale
    temp_term = np.exp(-((temp - t_center) ** 2) / (2.0 * sig_t ** 2 + 1e-9))
    ph_term = np.exp(-((pH - ph_center) ** 2) / (2.0 * sig_ph ** 2 + 1e-9))

    # Density term on raw scale
    dens_term = 1.0 + gamma * b

    # Interaction terms (centered for numerical stability)
    b_c = b - 3.0
    s_c = s - 3.0
    t_c = temp - 20.0
    ph_c = pH - 7.0
    
    interaction = beta_bs * (b_c * s_c) + beta_TpH * (t_c * ph_c)

    # Combine: multiplicative core + additive interactions
    result = C * monod * temp_term * ph_term * dens_term + interaction + bias

    # Ensure finite
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
