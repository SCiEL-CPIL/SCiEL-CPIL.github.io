import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a Chebyshev polynomial series.
    The basis functions are T_k(x/2) for k=0..9, which are orthogonal
    on the domain [-1, 1] corresponding to x in [-2, 2].
    This structure allows the model to approximate a wide range of
    smooth functions better than the linear seed.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # x is expected to be a 1D or 2D array. We reshape to 1D for processing
    # and reshape back at the end.
    original_shape = x.shape
    x_flat = x.flatten()
    
    # Scale x to [-1, 1] for Chebyshev polynomials
    t = x_flat / 2.0
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
    
    # We will build a basis matrix B of shape (N, 10)
    N = len(x_flat)
    B = np.empty((N, 10), dtype=float)
    
    B[:, 0] = 1.0
    if N > 0:
        B[:, 1] = t
        for k in range(2, 10):
            B[:, k] = 2.0 * t * B[:, k-1] - B[:, k-2]
    
    # Normalize each basis column to unit L2 norm for better conditioning
    # This makes the Gram matrix closer to identity, helping BFGS convergence
    col_norms = np.linalg.norm(B, axis=0)
    col_norms[col_norms < 1e-12] = 1.0  # Avoid division by zero
    B = B / col_norms
    
    # Prediction is B @ params
    # Ensure params is 1D
    p = params.flatten()
    
    # If the optimization hasn't finished or if we hit the deadline, 
    # we still return the current prediction. The evaluator calls this 
    # many times during BFGS steps. We just need to be fast.
    
    y = B @ p
    
    return y.reshape(original_shape)
# EVOLVE-BLOCK-END
