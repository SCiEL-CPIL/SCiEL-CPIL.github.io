"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of Gaussian Radial Basis Functions (RBFs) to approximate the target.
    Uses 10 fixed centers evenly spaced in [-2, 2] with a fixed width.
    This basis is effective for capturing local features and sharp transitions.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit

    p = params
    
    # Ensure finite params to avoid NaN/Inf during optimization steps
    if not np.all(np.isfinite(p)):
        p = np.ones(10)

    # Define fixed RBF centers and width
    # Centers evenly spaced in [-2, 2]
    centers = np.linspace(-1.8, 1.8, 10)
    # Width sigma chosen to provide overlap but maintain locality
    sigma = 0.6
    
    # Compute RBF basis: phi_i(x) = exp(-(x - c_i)^2 / (2 * sigma^2))
    # x is (N,), centers is (10,)
    # We want phi to be (10, N) for dot product with p (10,)
    # x[:, None] is (N, 1), centers[None, :] is (1, 10)
    # diff is (N, 10)
    diff = x[:, None] - centers[None, :]
    
    # Compute exponent
    # Clip to avoid overflow in exp, though with sigma=0.6 and x in [-2,2], 
    # max diff is 3.8, (3.8/0.6)^2 / 2 ~ 19, exp(-19) is small but safe.
    # However, during optimization, params might be large, but basis is independent of params.
    exponent = -0.5 * (diff / sigma) ** 2
    
    # Compute basis matrix
    # Phi is (N, 10)
    Phi = np.exp(exponent)
    
    # Compute prediction: sum(p_i * phi_i(x))
    # p is (10,), Phi is (N, 10)
    # pred = Phi @ p
    pred = Phi @ p
    
    # Check deadline (unlikely to be hit for simple vectorized ops, but good practice)
    if time.time() > deadline:
        # If we somehow exceeded time, return best effort (which is already computed)
        pass
        
    return pred
# EVOLVE-BLOCK-END
