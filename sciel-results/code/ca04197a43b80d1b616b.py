"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature."""
    # Improved model:
    # 1. Base linear terms: offset, strain slope, temp slope
    # 2. 2 Strain ramps: 2 breakpoints, 2 slopes
    # 3. 2 Temp ramps: 2 breakpoints, 1 shared slope (sum of ramps)
    # Total: 3 (base) + 4 (strain) + 3 (temp) = 10 params
    
    a0 = params[0]
    a1 = params[1]
    a2 = params[2]
    
    c1 = params[3]
    c2 = params[4]
    s1 = params[5]
    s2 = params[6]
    
    d1 = params[7]
    d2 = params[8]
    t1 = params[9]
    
    eps = 1e-6
    
    def smooth_ramp(x):
        return (x + np.sqrt(x**2 + eps)) / 2.0
    
    # Strain ramps
    r1 = smooth_ramp(strain - c1)
    r2 = smooth_ramp(strain - c2)
    
    # Temp ramps
    t_r1 = smooth_ramp(temp - d1)
    t_r2 = smooth_ramp(temp - d2)
    
    # Prediction
    # Note: t1 applies to both temp ramps, creating a piecewise linear temp effect with 2 kinks
    pred = a0 + a1 * strain + a2 * temp + s1 * r1 + s2 * r2 + t1 * (t_r1 + t_r2)
    
    return pred
# EVOLVE-BLOCK-END
