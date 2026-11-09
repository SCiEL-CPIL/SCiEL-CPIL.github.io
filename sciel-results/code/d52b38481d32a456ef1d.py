"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using piecewise-linear hinge basis functions.
    
    This creates a convex, linear-in-parameters model that captures local kinks
    and physiological thresholds (substrate saturation, temperature limits) without
    the local minima traps of multiplicative biological models.
    
    Basis functions:
    - Linear terms for each input
    - Hinge functions max(0, x-c) at key physiological thresholds
    - Product terms for key interactions
    """
    p = params
    
    # Centered inputs for numerical stability
    b_c = b - 3.0
    s_c = s - 10.0
    t_c = temp - 30.0
    p_c = pH - 7.0
    
    # Hinge functions for substrate (saturation threshold)
    s_hinge1 = np.maximum(0.0, s - 5.0)    # Low substrate region
    s_hinge2 = np.maximum(0.0, s - 15.0)   # High substrate region
    
    # Hinge functions for temperature (optimal range)
    t_hinge1 = np.maximum(0.0, temp - 25.0)  # Below optimal
    t_hinge2 = np.maximum(0.0, temp - 35.0)  # Above optimal
    
    # Hinge functions for pH (optimal range)
    ph_hinge1 = np.maximum(0.0, pH - 6.0)   # Acidic side
    ph_hinge2 = np.maximum(0.0, pH - 8.0)   # Alkaline side
    
    # Hinge functions for biomass density
    b_hinge1 = np.maximum(0.0, b - 2.0)     # Low density
    b_hinge2 = np.maximum(0.0, b - 4.0)     # High density (inhibition)
    
    # Key interaction terms (product of centered inputs)
    bs_int = b_c * s_c
    st_int = s_c * t_c
    sp_int = s_c * p_c
    bt_int = b_c * t_c
    
    # Construct prediction as linear combination of basis functions
    pred = (
        p[0] * b_c +
        p[1] * s_c +
        p[2] * t_c +
        p[3] * p_c +
        p[4] * s_hinge1 +
        p[5] * s_hinge2 +
        p[6] * t_hinge1 +
        p[7] * t_hinge2 +
        p[8] * ph_hinge1 +
        p[9] * ph_hinge2
    )
    
    # Add interaction terms if we have more params (but we're limited to 10)
    # For now, use the 10 params for the most important terms
    
    return pred
# EVOLVE-BLOCK-END
