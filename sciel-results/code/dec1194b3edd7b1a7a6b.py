"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for an aluminium rod.
    
    Uses a linear-in-parameters basis function approach for robust BFGS optimization.
    The basis functions capture:
    - Elastic linear region (strain)
    - Plastic hardening (quadratic/cubic strain terms)
    - Temperature dependence (linear temp, interactions)
    - Non-linear strain effects (higher order terms)
    
    This linear structure ensures the MSE loss is convex in params,
    allowing BFGS to reliably find the global minimum.
    """
    s = strain
    t = temp
    p = params
    
    # Normalize inputs to improve numerical conditioning
    # Training ranges: strain [-0.001911, 0.9636], temp [0.06667, 1.0]
    s_min = -0.001911
    s_max = 0.9636
    t_min = 0.06667
    t_max = 1.0
    
    s_range = s_max - s_min
    t_range = t_max - t_min
    
    s_norm = (s - s_min) / s_range
    t_norm = (t - t_min) / t_range
    
    # Clip to [0, 1] for stability during optimization
    s_norm = np.clip(s_norm, 0.0, 1.0)
    t_norm = np.clip(t_norm, 0.0, 1.0)
    
    # Basis functions:
    # 0: Intercept
    # 1: Linear strain (elastic)
    # 2: Linear temp
    # 3: Strain*temp interaction
    # 4: Quadratic strain (plastic hardening onset)
    # 5: Quadratic strain * temp
    # 6: Cubic strain (plastic hardening)
    # 7: Cubic strain * temp
    # 8: Quartic strain (saturation behavior)
    # 9: Quartic strain * temp
    
    phi0 = np.ones_like(s_norm)
    phi1 = s_norm
    phi2 = t_norm
    phi3 = s_norm * t_norm
    phi4 = s_norm * s_norm
    phi5 = phi4 * t_norm
    phi6 = phi4 * s_norm
    phi7 = phi6 * t_norm
    phi8 = phi6 * s_norm
    phi9 = phi8 * t_norm
    
    # Linear combination of basis functions
    result = (p[0] * phi0 + 
              p[1] * phi1 + 
              p[2] * phi2 + 
              p[3] * phi3 + 
              p[4] * phi4 + 
              p[5] * phi5 + 
              p[6] * phi6 + 
              p[7] * phi7 + 
              p[8] * phi8 + 
              p[9] * phi9)
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return result
# EVOLVE-BLOCK-END
