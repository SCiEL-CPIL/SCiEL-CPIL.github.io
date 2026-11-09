"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Piecewise linear (hat function) basis for symbolic regression.
    
    This structural change moves away from smooth global basis functions
    (polynomials, trig, exp, RBFs) to a piecewise linear basis.
    Hat functions can approximate sharp transitions, discontinuities in derivative,
    and non-smooth features that smooth bases struggle with, while remaining
    linear-in-parameters for stable BFGS convergence.
    
    We use 10 hat functions with knots evenly spaced across [0.1, 6].
    Each hat function is a triangular bump: 0 outside [k_i - w, k_i + w],
    rising linearly to 1 at k_i, then falling linearly to 0.
    
    This creates a continuous but non-smooth function class.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Define 10 knot positions evenly spaced in [0.1, 6]
    # Using linspace to spread them across the domain
    knots = np.linspace(0.5, 5.5, 10)
    
    # Half-width for each hat function
    # Spacing between knots is (5.5 - 0.5) / 9 = 5.0 / 9 ≈ 0.556
    # Use half the spacing so hats don't overlap too much, but still cover domain
    spacing = (knots[-1] - knots[0]) / (len(knots) - 1)
    half_width = spacing * 0.6  # slight overlap for smoother transitions
    
    # Build basis matrix: each column is one hat function
    basis = np.zeros((len(x), len(knots)))
    
    for i, k in enumerate(knots):
        left = k - half_width
        right = k + half_width
        # Mask for points within the hat's support
        mask = (x >= left) & (x <= right)
        if np.any(mask):
            # Rising edge: linearly increase from 0 to 1
            rising_mask = mask & (x < k)
            if np.any(rising_mask):
                basis[rising_mask, i] = (x[rising_mask] - left) / (k - left)
            # Falling edge: linearly decrease from 1 to 0
            falling_mask = mask & (x >= k)
            if np.any(falling_mask):
                basis[falling_mask, i] = (right - x[falling_mask]) / (right - k)
    
    # Compute linear combination
    y = basis @ params
    
    return y

# EVOLVE-BLOCK-END
