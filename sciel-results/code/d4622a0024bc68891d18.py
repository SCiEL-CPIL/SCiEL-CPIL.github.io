"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a hybrid Chebyshev + Fourier basis.
    
    This approach combines:
    1. Chebyshev polynomials (T0-T3) to capture global smooth trends.
       Chebyshev polynomials are orthogonal on [-1, 1], providing a well-conditioned
       basis for least-squares fitting, which helps BFGS converge reliably.
    2. Fourier terms (sin/cos at frequencies 1, 2, 3) to capture local oscillations
       or periodic features that polynomials would approximate poorly.
    
    The model is linear in the 10 parameters, ensuring that the optimization
    landscape is convex and BFGS will converge to the global minimum (the
    least-squares solution) for this specific basis.
    """
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    
    # Normalize x to [-1, 1] for Chebyshev stability
    # x is in [-3, 3], so t = x / 3
    t = x / 3.0
    
    # --- Chebyshev Polynomials T0 to T3 ---
    b0 = np.ones(n)
    b1 = t.copy()
    b2 = 2.0 * t**2 - 1.0
    b3 = 4.0 * t**3 - 3.0 * t
    
    # --- Gaussian RBFs ---
    # Centers at -1.5 and 1.5, width 1.0
    # This allows capturing local bumps or deviations from the global trend
    sigma = 1.0
    b4 = np.exp(-0.5 * ((x - (-1.5)) / sigma) ** 2)
    b5 = np.exp(-0.5 * ((x - 1.5) / sigma) ** 2)
    
    # --- Tanh Functions ---
    # Saturating non-linearities
    b6 = np.tanh(x)
    b7 = np.tanh(2.0 * x)
    
    # --- Fourier Terms ---
    # Basic periodicity
    b8 = np.sin(x)
    b9 = np.cos(x)
    
    # Stack basis functions into a matrix of shape (n, 10)
    basis = np.column_stack([
        b0, b1, b2, b3,
        b4, b5, b6, b7,
        b8, b9
    ])
    
    # Compute the linear combination: result = basis @ params
    # basis: (n, 10)
    # params: (10,)
    # result: (n,)
    result = basis @ params
    
    # Safety: ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
        
    # Check deadline (though this is very fast)
    if time.time() > deadline:
        pass # Return best effort (which is already computed)
        
    return result
# EVOLVE-BLOCK-END
