"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression equation using Chebyshev polynomials of the first kind.
    Maps x in [-3, 3] to u in [-1, 1] and evaluates a linear combination of
    T_0(u) through T_9(u) with 10 parameters.
    """
    # Convert to float64 array
    x = np.asarray(x, dtype=np.float64)
    
    # Scale input to [-1, 1] for Chebyshev polynomials
    u = x / 3.0
    
    # Clamp to [-1, 1] to avoid numerical issues (though x is uniform [-3,3])
    u_clamped = np.clip(u, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(u) = 1
    # T_1(u) = u
    # T_n(u) = 2*u*T_{n-1}(u) - T_{n-2}(u)
    
    t0 = np.ones_like(u_clamped)
    t1 = u_clamped.copy()
    
    # Start with the first two terms
    result = params[0] * t0 + params[1] * t1
    
    # Add higher-order Chebyshev terms
    if len(params) > 2:
        t_prev2 = t0
        t_prev1 = t1
        for k in range(2, 10):
            # T_k = 2*u*T_{k-1} - T_{k-2}
            t_curr = 2.0 * u_clamped * t_prev1 - t_prev2
            result += params[k] * t_curr
            t_prev2 = t_prev1
            t_prev1 = t_curr
    
    # Ensure finite output; fallback to linear if non-finite
    if not np.all(np.isfinite(result)):
        result = np.clip(params[0], -1e6, 1e6) + np.clip(params[1], -1e6, 1e6) * u_clamped
    
    return result
# EVOLVE-BLOCK-END
