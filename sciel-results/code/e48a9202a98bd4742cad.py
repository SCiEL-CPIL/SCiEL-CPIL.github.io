"""
Symbolic regression: recover an unknown function f(x) with 1 input from [-3, 3].
Strategy: Hybrid Global-Local Basis.
- Global: Chebyshev polynomials T0-T2 to capture the general trend efficiently.
- Local: 7 Gaussian bumps centered at fixed locations in [-3, 3].
The model is linear in parameters, so BFGS converges to the global least-squares solution.
The local Gaussians allow the model to fit local features (bumps, drops, wiggles) 
that global polynomials cannot capture without increasing degree (and ill-conditioning).
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model with 10 parameters.
    
    Basis functions:
    0: 1 (intercept)
    1: Chebyshev T1(x/3)
    2: Chebyshev T2(x/3)
    3-9: Gaussian bumps exp(-(x-c_i)^2 / sigma^2)
    
    Centers c_i are evenly spaced in [-3, 3].
    Sigma is chosen to cover the interval with overlap but remain local.
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    orig_shape = x.shape
    x_flat = x.ravel()
    
    # Global trend: Chebyshev polynomials on scaled domain [-1, 1]
    t = x_flat / 3.0
    b0 = np.ones_like(t)
    b1 = t
    b2 = 2.0 * t**2 - 1.0
    
    # Local features: Gaussian bumps
    # 7 centers evenly spaced from -3 to 3
    centers = np.linspace(-3.0, 3.0, 7)
    # Sigma = 0.8 provides good overlap and localization
    sigma2 = 0.8**2
    
    result = p[0] * b0 + p[1] * b1 + p[2] * b2
    
    for i in range(7):
        c = centers[i]
        g = np.exp(-(x_flat - c)**2 / sigma2)
        result += p[3 + i] * g
        
    return result.reshape(orig_shape)
# EVOLVE-BLOCK-END
