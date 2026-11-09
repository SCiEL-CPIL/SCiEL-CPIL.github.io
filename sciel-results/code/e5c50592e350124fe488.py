"""Symbolic regression with 3 inputs using a hybrid Chebyshev + RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining global Chebyshev polynomial terms with local RBF terms.
    
    Parameters (10 total):
    - p[0]: constant offset
    - p[1]: coefficient for T_1(x) * T_1(y)
    - p[2]: coefficient for T_1(x) * T_1(z)
    - p[3]: coefficient for T_1(y) * T_1(z)
    - p[4]: coefficient for T_2(x) * T_2(y) * T_2(z) (high-order interaction)
    - p[5]: center_x for RBF
    - p[6]: center_y for RBF
    - p[7]: center_z for RBF
    - p[8]: amplitude for RBF
    - p[9]: width parameter for RBF (positive)
    
    The Chebyshev terms capture global smooth structure, while the RBF term
    captures local deviations or sharp features that global polynomials miss.
    """
    p = params
    
    # Map inputs from [0.5, 2] to [-1, 1] for Chebyshev stability
    tx = (4.0 * x - 5.0) / 3.0
    ty = (4.0 * y - 5.0) / 3.0
    tz = (4.0 * z - 5.0) / 3.0
    
    # Chebyshev basis terms
    # T_1(t) = t
    # T_2(t) = 2*t^2 - 1
    
    T1x = tx
    T1y = ty
    T1z = tz
    
    T2x = 2.0 * tx * tx - 1.0
    T2y = 2.0 * ty * ty - 1.0
    T2z = 2.0 * tz * tz - 1.0
    
    # Global interaction terms
    term_xy = T1x * T1y
    term_xz = T1x * T1z
    term_yz = T1y * T1z
    term_xyz = T2x * T2y * T2z  # High-order interaction for complex structure
    
    # RBF term: Gaussian bump at (p[5], p[6], p[7]) with amplitude p[8] and width related to p[9]
    cx = p[5]
    cy = p[6]
    cz = p[7]
    amp = p[8]
    
    # Width: use abs to ensure positive, add epsilon to avoid division by zero
    # If p[9] is large, width is large (wide bump). If small, narrow bump.
    # Let's map p[9] to width directly, ensuring it's positive.
    w = np.maximum(np.abs(p[9]), 1e-6)
    
    dx = x - cx
    dy = y - cy
    dz = z - cz
    dist_sq = dx * dx + dy * dy + dz * dz
    
    rbf_term = amp * np.exp(-dist_sq / (2.0 * w * w))
    
    # Combine all terms
    result = (
        p[0] +
        p[1] * term_xy +
        p[2] * term_xz +
        p[3] * term_yz +
        p[4] * term_xyz +
        rbf_term
    )
    
    return result
# EVOLVE-BLOCK-END
