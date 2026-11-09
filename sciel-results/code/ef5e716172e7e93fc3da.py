"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis combining Chebyshev polynomials (global stability) with
    Gaussian bumps (local flexibility) and oscillatory terms (frequency capture).
    
    Uses 4 Chebyshev polynomials T_0..T_3 on scaled domain [-1,1],
    4 Gaussian bumps centered at strategic points, and 2 oscillatory terms.
    This provides a well-conditioned global approximation with local adaptability.
    """
    # Scale input from [-2, 2] to [-1, 1] for Chebyshev evaluation
    t = x / 2.0
    
    # Chebyshev polynomials T_0 through T_3 using recurrence
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    
    # Gaussian basis functions centered at strategic points in [-2, 2]
    # Centers: -1.5, -0.5, 0.5, 1.5 with width ~0.8
    centers = np.array([-1.5, -0.5, 0.5, 1.5])
    width = 0.8
    
    x_expanded = x[:, np.newaxis]
    gaussians = np.exp(-((x_expanded - centers) ** 2) / (2.0 * width ** 2))
    # gaussians shape: (n_points, 4)
    
    # Oscillatory terms for additional frequency capture
    osc1 = np.sin(np.pi * x / 2.0)
    osc2 = np.cos(np.pi * x / 2.0)
    
    # Combine all basis terms: 4 Chebyshev + 4 Gaussians + 2 oscillatory = 10 total
    # params[0:4] for Chebyshev, params[4:8] for Gaussians, params[8:10] for oscillatory
    
    result = (T0 * params[0] +
              T1 * params[1] +
              T2 * params[2] +
              T3 * params[3] +
              gaussians[:, 0] * params[4] +
              gaussians[:, 1] * params[5] +
              gaussians[:, 2] * params[6] +
              gaussians[:, 3] * params[7] +
              osc1 * params[8] +
              osc2 * params[9])
    
    return result
# EVOLVE-BLOCK-END
