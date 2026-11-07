"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis for symbolic regression: Chebyshev polynomials (well-conditioned
    on the domain) plus selected nonlinear features (exp, log, trig, reciprocal)
    to capture common functional forms.
    
    The Chebyshev part provides a stable, well-conditioned backbone for BFGS.
    The additional nonlinear features expand the model class beyond polynomials.
    
    Basis functions (10 total):
    0: T_0(t) = 1
    1: T_1(t) = t
    2: T_2(t)
    3: T_3(t)
    4: T_4(t)
    5: exp(x)
    6: exp(-x)
    7: log(x + 1)
    8: sin(x)
    9: 1/x
    
    where t = (2x - (x_max + x_min)) / (x_max - x_min) maps [0.1, 6] to [-1, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x to t in [-1, 1] for Chebyshev polynomials
    x_min = 0.1
    x_max = 6.0
    t = (2.0 * x - (x_max + x_min)) / (x_max - x_min)
    t = np.clip(t, -1.0, 1.0)
    
    n_points = x.shape[0]
    
    # Compute Chebyshev polynomials T_0 to T_4
    basis = np.empty((n_points, 10), dtype=np.float64)
    basis[:, 0] = 1.0
    basis[:, 1] = t
    
    for k in range(2, 5):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
    
    # Standardize Chebyshev columns for better conditioning
    # For T_k(t) with t ~ Uniform[-1, 1]: mean=0, var=1 for k=0, var=0.5 for k>=1
    std_cheb = np.ones(5, dtype=np.float64)
    std_cheb[1:] = np.sqrt(0.5)
    
    for k in range(5):
        basis[:, k] /= std_cheb[k]
    
    # Nonlinear features
    x_safe = np.clip(x, 1e-10, None)
    basis[:, 5] = np.exp(x)
    basis[:, 6] = np.exp(-x)
    basis[:, 7] = np.log(x_safe + 1.0)
    basis[:, 8] = np.sin(x)
    basis[:, 9] = 1.0 / x_safe
    
    # Standardize nonlinear features using approximate moments for x ~ Uniform[0.1, 6]
    # Compute means and stds numerically for robustness
    # Precompute using a fine grid approximation
    x_grid = np.linspace(0.1, 6.0, 1000)
    for k in range(5, 10):
        col = basis[x_grid.shape[0] and 0:0]  # placeholder
        # Actually compute from x_grid
        if k == 5:
            col = np.exp(x_grid)
        elif k == 6:
            col = np.exp(-x_grid)
        elif k == 7:
            col = np.log(x_grid + 1.0)
        elif k == 8:
            col = np.sin(x_grid)
        elif k == 9:
            col = 1.0 / x_grid
        
        mean_k = np.mean(col)
        std_k = np.std(col)
        if std_k < 1e-10:
            std_k = 1.0
        basis[:, k] = (basis[:, k] - mean_k) / std_k
    
    # Combine with parameters
    return basis @ params
# EVOLVE-BLOCK-END
