"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """E. coli growth rate from population density, substrate, temperature and pH.
    
    Uses a generalized additive model (GAM) with B-spline basis functions for each input.
    The 10 parameters are coefficients for these bases, making the optimization convex
    and easy for BFGS to solve.
    """
    # Normalize inputs to [0, 1] based on training ranges
    # b: [1, 4.998] -> [0, 1]
    b_norm = np.clip((b - 1.0) / (4.998 - 1.0), 0, 1)
    # s: [1, 4.999] -> [0, 1]
    s_norm = np.clip((s - 1.0) / (4.999 - 1.0), 0, 1)
    # temp: [1.002, 39.98] -> [0, 1]
    temp_norm = np.clip((temp - 1.002) / (39.98 - 1.002), 0, 1)
    # pH: [2, 12] -> [0, 1]
    pH_norm = np.clip((pH - 2.0) / (12.0 - 2.0), 0, 1)
    
    # Create B-spline-like basis functions (using polynomial bases for simplicity and convexity)
    # For each variable, use 2 basis functions (linear and quadratic)
    # Total: 4 variables * 2 bases = 8 parameters, plus 2 for interactions or intercept
    
    # Basis functions for b
    b_basis0 = b_norm
    b_basis1 = b_norm ** 2
    
    # Basis functions for s
    s_basis0 = s_norm
    s_basis1 = s_norm ** 2
    
    # Basis functions for temp
    temp_basis0 = temp_norm
    temp_basis1 = temp_norm ** 2
    
    # Basis functions for pH
    pH_basis0 = pH_norm
    pH_basis1 = pH_norm ** 2
    
    # Use 10 parameters: 8 for main effects (2 per variable) + 2 for key interactions
    # Interaction 1: temp * pH (temperature and pH often interact)
    interaction1 = temp_basis0 * pH_basis0
    
    # Interaction 2: s * b (substrate and population density)
    interaction2 = s_basis0 * b_basis0
    
    # Combine all terms
    prediction = (
        params[0] * b_basis0 +
        params[1] * b_basis1 +
        params[2] * s_basis0 +
        params[3] * s_basis1 +
        params[4] * temp_basis0 +
        params[5] * temp_basis1 +
        params[6] * pH_basis0 +
        params[7] * pH_basis1 +
        params[8] * interaction1 +
        params[9] * interaction2
    )
    
    return prediction
# EVOLVE-BLOCK-END
