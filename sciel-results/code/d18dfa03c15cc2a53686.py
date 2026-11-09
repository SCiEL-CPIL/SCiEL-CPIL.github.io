import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate: Separable multiplicative basis.
    
    This model uses a basis of products of normalized inputs to capture
    multiplicative coupling between variables. The basis consists of the
    8 terms from the tensor product of linear bases in x, y, z, plus two
    additional quadratic terms (u^2, v^2) to utilize the 10-parameter budget
    and allow for slight deviations from strict separability.
    
    Parameters:
    p[0]: constant
    p[1]: u
    p[2]: v
    p[3]: w
    p[4]: u*v
    p[5]: u*w
    p[6]: v*w
    p[7]: u*v*w
    p[8]: u^2
    p[9]: v^2
    
    Inputs are normalized to [0, 1] for numerical stability.
    """
    p = params
    
    # Normalize inputs from [0.5, 2.0] to [0, 1]
    # This ensures all basis terms are in [0, 1], improving conditioning
    u = (x - 0.5) / 1.5
    v = (y - 0.5) / 1.5
    w = (z - 0.5) / 1.5
    
    # Construct the separable multiplicative basis
    # Terms: 1, u, v, w, uv, uw, vw, uvw
    # Plus extra terms: u^2, v^2
    
    result = (p[0] * 1.0 +
              p[1] * u +
              p[2] * v +
              p[3] * w +
              p[4] * u * v +
              p[5] * u * w +
              p[6] * v * w +
              p[7] * u * v * w +
              p[8] * u * u +
              p[9] * v * v)
    
    return result
