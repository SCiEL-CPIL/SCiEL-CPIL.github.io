"""Combined model: Bivariate rational function with normalized inputs."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a bivariate rational function model with 10 fitted parameters.
    Inputs are normalized to improve conditioning for the BFGS optimizer.
    
    Model:
    s = (strain - 0.48) / 0.48
    t = (temp - 0.53) / 0.47
    
    sigma = (p0 + p1*s + p2*s^2 + p3*t + p4*t^2 + p5*s*t)
            / (1 + p6*s + p7*s^2 + p8*t + p9*s*t)
    """
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to roughly [-1, 1] range for better conditioning
    # strain in [-0.001911, 0.9636] -> center ~0.48, scale ~0.48
    s = (strain - 0.48) / 0.48
    # temp in [0.06667, 1] -> center ~0.53, scale ~0.47
    t = (temp - 0.53) / 0.47
    
    # Numerator: flexible polynomial in normalized coords
    num = (p0 + p1 * s + p2 * s**2 + 
           p3 * t + p4 * t**2 + 
           p5 * s * t)
    
    # Denominator: 1 + small perturbation terms (keep near 1 for stability)
    den = 1.0 + p6 * s + p7 * s**2 + p8 * t + p9 * s * t
    
    # Safety: clamp denominator away from zero to avoid numerical instability
    # Using a slightly larger epsilon (0.01) than 1e-10 to avoid extreme gradients
    # but small enough to not distort the function shape significantly.
    # Inspiration 1 used 0.05. Let's try 0.01 for more fidelity.
    den_safe = np.where(np.abs(den) < 0.01, np.sign(den + 1e-300) * 0.01, den)
    
    stress = num / den_safe
    
    return stress
# EVOLVE-BLOCK-END
