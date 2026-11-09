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
    polynomial basis cannot represent.
    
    Basis (10 params total):
    - 4 global terms: bias, x, v, x*v (core oscillator dynamics)
    - 6 RBF terms: Gaussian bumps in (x,v) space at strategic centers
    
    RBF centers are chosen at representative points in the training range:
    x in [-0.2056, 0.1749], v in [-0.3391, 0.3624]
    
    The RBF width is fixed to provide good local coverage without extreme
    ill-conditioning. Centers are spread across the state space.
    """
    
    t = np.asarray(t)
    x = np.asarray(v)  # placeholder, will fix below
    x = np.asarray(x)
    v = np.asarray(v)
    
    # Global physics-motivated terms (4 params)
    g1 = np.ones_like(x)       # bias
    g2 = x                      # linear spring
    g3 = v                      # linear damping
    g4 = x * v                  # linear coupling
    
    # RBF centers in (x, v) space - spread across training range
    # x range: [-0.2056, 0.1749], v range: [-0.3391, 0.3624]
    centers = np.array([
        [-0.10, -0.15],
        [0.05, 0.15],
        [-0.05, 0.05],
        [0.10, -0.10],
        [-0.15, 0.20],
        [0.00, -0.20],
    ])
    
    # RBF width - chosen to give local coverage
    sigma = 0.15
    
    # Compute RBF features (6 params)
    rbf_features = []
    for cx, cv in centers:
        dx = x - cx
        dv = v - cv
        r2 = dx * dx + dv * dv
        rbf = np.exp(-r2 / (2.0 * sigma * sigma))
        rbf_features.append(rbf)
    
    # Construct full design matrix (N, 10)
    features = np.column_stack((g1, g2, g3, g4) + tuple(rbf_features))
    
    # Linear combination with params
    return features @ params

# EVOLVE-BLOCK-END
