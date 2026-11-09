"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid Global-Local Nonlinear Oscillator Model.
    
    Structural change: Replace the pure polynomial/trig global basis with a hybrid
    that combines a small number of physics-motivated global terms with localized
    Gaussian RBF terms in the (x, v) state space. This allows the model to capture
    both the overall linear structure AND local nonlinear features that a global
    polynomial basis cannot represent well.
    
    Basis (10 terms):
    1. Bias (1)
    2. sin(t) - periodic drive
    3. x - linear spring
    4. v - linear damping
    5-10. Six Gaussian RBFs centered at strategically chosen (x, v) points
         to capture local nonlinear behavior in the state space.
    
    The RBF centers are placed at:
    - (0, 0): origin - captures equilibrium behavior
    - (0.15, 0): positive x displacement
    - (-0.15, 0): negative x displacement  
    - (0, 0.3): positive velocity
    - (0, -0.3): negative velocity
    - (0.15, 0.3): combined positive state
    
    Width sigma=0.25 is chosen to be on the scale of the input ranges.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    # Global terms
    b1 = np.ones_like(t)
    b2 = np.sin(t)
    b3 = x
    b4 = v
    
    # RBF centers in (x, v) space
    # Chosen to cover the input range: x in [-0.2056, 0.1749], v in [-0.3391, 0.3624]
    centers = np.array([
        [0.0, 0.0],
        [0.15, 0.0],
        [-0.15, 0.0],
        [0.0, 0.3],
        [0.0, -0.3],
        [0.15, 0.3]
    ])
    
    sigma = 0.25  # RBF width
    
    # Compute RBF values: for each center, compute exp(-||[x,v] - center||^2 / (2*sigma^2))
    # x, v are shape (N,), centers are shape (6, 2)
    dx = x[:, np.newaxis] - centers[0, :]  # (N, 1)
    dv = v[:, np.newaxis] - centers[1, :]  # (N, 1)
    
    # Actually, let's compute all 6 RBFs properly
    b5 = np.exp(-((x - 0.0)**2 + (v - 0.0)**2) / (2.0 * sigma**2))
    b6 = np.exp(-((x - 0.15)**2 + (v - 0.0)**2) / (2.0 * sigma**2))
    b7 = np.exp(-((x - (-0.15))**2 + (v - 0.0)**2) / (2.0 * sigma**2))
    b8 = np.exp(-((x - 0.0)**2 + (v - 0.3)**2) / (2.0 * sigma**2))
    b9 = np.exp(-((x - 0.0)**2 + (v - (-0.3))**2) / (2.0 * sigma**2))
    b10 = np.exp(-((x - 0.15)**2 + (v - 0.3)**2) / (2.0 * sigma**2))
    
    features = np.column_stack((
        b1, b2, b3, b4, b5, b6, b7, b8, b9, b10
    ))
    
    return features @ params

# EVOLVE-BLOCK-END
