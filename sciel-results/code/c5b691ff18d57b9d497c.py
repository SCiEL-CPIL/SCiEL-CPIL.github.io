import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a Chebyshev polynomial series.
    The basis functions are T_k(x/2) for k=0..9.
    Chebyshev polynomials are orthogonal on the interval [-1, 1], which corresponds
    to the input domain x in [-2, 2]. This orthogonality ensures a well-conditioned
    system, facilitating stable convergence for the BFGS optimizer.
    """
    original_shape = x.shape
    x_flat = x.flatten()
    N = len(x_flat)
    
    if N == 0:
        return np.empty(original_shape)
    
    # Scale x to [-1, 1]
    t = x_flat / 2.0
    
    # Build basis matrix (N, 10)
    B = np.empty((N, 10), dtype=float)
    
    # T_0(t) = 1
    B[:, 0] = 1.0
    # T_1(t) = t
    B[:, 1] = t
    
    # Compute T_2 to T_9 using the recurrence relation
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    for k in range(2, 10):
        B[:, k] = 2.0 * t * B[:, k-1] - B[:, k-2]
        
    p = params.flatten()
    y = B @ p
    
    return y.reshape(original_shape)
