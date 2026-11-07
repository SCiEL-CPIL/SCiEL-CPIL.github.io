"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a linear combination of basis functions.
    
    We use a Fourier-like basis (sines and cosines) centered and scaled to the
    input domain [0.1, 6]. This provides a smooth, periodic-capable basis that
    BFGS can optimize robustly from the all-ones starting point.
    
    The 10 parameters correspond to coefficients for:
      1. constant
      2. x (linear)
      3. sin(2*pi*x)
      4. cos(2*pi*x)
      5. sin(2*pi*x/2)  (i.e., sin(pi*x))
      6. cos(2*pi*x/2)  (i.e., cos(pi*x))
      7. sin(2*pi*x/3)
      8. cos(2*pi*x/3)
      9. sin(2*pi*x/4)
      10. cos(2*pi*x/4)
    
    This basis spans a wide range of smooth functions on [0.1, 6].
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    # Build the basis matrix
    # Using frequencies that capture features on the scale of the domain
    # Domain is [0.1, 6], width ~5.9
    # Frequencies: 1, 2, 3, 4 (in units of 2*pi per unit x)
    # But to be more flexible, use scaled frequencies
    
    # Center and scale x for better conditioning
    # Map [0.1, 6] to roughly [0, 1] for frequency design
    x_centered = x - 3.05  # center of [0.1, 6]
    x_scaled = x_centered / 3.0  # scale to ~[-1, 1]
    
    # Basis functions using the scaled variable
    # This makes the basis well-conditioned for BFGS
    b1 = np.ones_like(x_scaled)
    b2 = x_scaled
    b3 = np.sin(np.pi * x_scaled)
    b4 = np.cos(np.pi * x_scaled)
    b5 = np.sin(2 * np.pi * x_scaled)
    b6 = np.cos(2 * np.pi * x_scaled)
    b7 = np.sin(3 * np.pi * x_scaled)
    b8 = np.cos(3 * np.pi * x_scaled)
    b9 = np.sin(4 * np.pi * x_scaled)
    b10 = np.cos(4 * np.pi * x_scaled)
    
    # Linear combination
    return (params[0] * b1 +
            params[1] * b2 +
            params[2] * b3 +
            params[3] * b4 +
            params[4] * b5 +
            params[5] * b6 +
            params[6] * b7 +
            params[7] * b8 +
            params[8] * b9 +
            params[9] * b10)
# EVOLVE-BLOCK-END
