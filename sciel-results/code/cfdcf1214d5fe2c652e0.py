"""Symbolic regression with 1 input."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a mixed basis of 10 terms.
    
    Combines Chebyshev polynomials (for smooth polynomial components) with
    trigonometric, hyperbolic, and exponential terms to cover a wider
    hypothesis space. This helps BFGS find good fits for diverse target
    functions, including those with periodic or saturating behavior.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [-3, 3] to t in [-1, 1] for stability
    t = x / 3.0
    t_safe = np.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Build a diverse basis of 10 terms:
    # 0: T_0 = 1 (constant)
    # 1: T_1 = t (linear)
    # 2: T_2 = 2t^2 - 1 (quadratic Chebyshev)
    # 3: T_3 = 4t^3 - 3t (cubic Chebyshev)
    # 4: sin(pi*t) - periodic component
    # 5: cos(pi*t) - periodic component
    # 6: tanh(2*t) - saturating nonlinearity
    # 7: exp(-t^2) - Gaussian bump
    # 8: T_5 = 16t^5 - 20t^3 + 5t (higher-order Chebyshev)
    # 9: sin(2*pi*t) - higher-frequency periodic
    
    basis = np.empty((10, len(t_safe)), dtype=np.float64)
    basis[0] = np.ones_like(t_safe)
    basis[1] = t_safe.copy()
    basis[2] = 2.0 * t_safe**2 - 1.0
    basis[3] = 4.0 * t_safe**3 - 3.0 * t_safe
    basis[4] = np.sin(np.pi * t_safe)
    basis[5] = np.cos(np.pi * t_safe)
    basis[6] = np.tanh(2.0 * t_safe)
    basis[7] = np.exp(-t_safe**2)
    # T_5 via recurrence: T_4 = 2t*T_3 - T_2, T_5 = 2t*T_4 - T_3
    T3 = basis[3]
    T2 = basis[2]
    T4 = 2.0 * t_safe * T3 - T2
    basis[8] = 2.0 * t_safe * T4 - T3
    basis[9] = np.sin(2.0 * np.pi * t_safe)
    
    # Build the linear combination
    result = np.zeros_like(t_safe)
    for k in range(10):
        result += params[k] * basis[k]
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
