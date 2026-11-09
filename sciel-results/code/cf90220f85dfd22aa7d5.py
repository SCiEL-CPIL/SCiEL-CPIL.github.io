"""E. coli growth rate model: flexible polynomial basis with sigmoid gating."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible growth rate model using a polynomial basis with soft gating.
    
    Instead of rigid biological priors, this uses a sum of polynomial terms
    in the inputs, allowing the optimizer to discover the functional form.
    A soft saturation (tanh) is applied to the output to keep it bounded.
    
    Parameters (10):
    p0: coefficient for 1
    p1: coefficient for s
    p2: coefficient for s^2
    p3: coefficient for temp
    p4: coefficient for temp^2
    p5: coefficient for pH
    p6: coefficient for pH^2
    p7: coefficient for b
    p8: scale for tanh saturation
    p9: bias offset
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Build polynomial terms
    term1 = p0 * np.ones_like(b)
    term2 = p1 * s
    term3 = p2 * s**2
    term4 = p3 * temp
    term5 = p4 * temp**2
    term6 = p5 * pH
    term7 = p6 * pH**2
    term8 = p7 * b
    
    # Sum of polynomial terms
    raw = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + p9
    
    # Apply soft saturation to keep outputs reasonable
    # p8 controls the sharpness; if p8 is large, tanh is near identity for small inputs
    # If p8 is small, outputs are compressed
    result = np.tanh(raw / (np.abs(p8) + 1e-6)) * (np.abs(p8) + 1e-6)
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
