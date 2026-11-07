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
    
    # Normalize to [0, 1] range
    s_norm = (s - s_min) / (s_max - s_min)
    t_norm = (t - t_min) / (t_max - t_min)
    
    # Clip to valid range for safety
    s_norm = np.clip(s_norm, 0, 1)
    t_norm = np.clip(t_norm, 0, 1)
    
    # Basis functions
    # 1. Constant term
    b0 = np.ones_like(s)
    
    # 2. Linear strain
    b1 = s_norm
    
    # 3. Quadratic strain (plastic hardening)
    b2 = s_norm ** 2
    
    # 4. Cubic strain
    b3 = s_norm ** 3
    
    # 5. Linear temperature
    b4 = t_norm
    
    # 6. Strain-temperature interaction
    b5 = s_norm * t_norm
    
    # 7. Temperature squared
    b6 = t_norm ** 2
    
    # 8. Strain squared times temperature
    b7 = s_norm ** 2 * t_norm
    
    # 9. Sigmoid-like transition for yield point
    # Using a smooth step function centered around typical yield strain
    # For aluminium, yield strain is typically around 0.001-0.005
    # In normalized coordinates, this is roughly (0.001 - (-0.001911)) / (0.9636 - (-0.001911)) ≈ 0.003
    yield_center = 0.003
    yield_width = 0.01
    b8 = 1.0 / (1.0 + np.exp(-100 * (s_norm - yield_center) / yield_width))
    
    # 10. Strain times sigmoid (post-yield hardening)
    b9 = s_norm * b8
    
    # Combine basis functions with parameters
    stress = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + 
              p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
              p[8] * b8 + p[9] * b9)
    
    return stress
# EVOLVE-BLOCK-END
