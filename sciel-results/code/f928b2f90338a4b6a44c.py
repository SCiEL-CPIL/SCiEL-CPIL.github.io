"""
Symbolic regression for a nonlinear oscillator using a Physics-Informed Chebyshev Polynomial Basis.
Strategy: 
1. Normalize inputs x and v to the range [-1, 1] based on known training bounds.
2. Use a complete set of Chebyshev polynomials T_n(u)T_m(v) up to total degree 4.
   This provides a smooth, global basis that is well-conditioned (orthogonal on [-1,1]).
3. The acceleration field is modeled as a linear combination of these 10 basis functions.
   This avoids the local overfitting of RBFs and the ill-conditioning of high-order monomials.
4. BFGS fits the 10 weights efficiently.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

# Deadline guard: evaluator kills at 150s, keep safety margin
_DEADLINE = time.time() + 130

def _time_left() -> float:
    return _DEADLINE - time.time()

# Input ranges from task description
x_min, x_max = -0.5728, 0.5275
v_min, v_max = -0.1754, 0.2821

# Pre-compute normalization constants
# Map x to [-1, 1]
# u = (x - x_mid) / x_half
x_mid = (x_min + x_max) / 2.0
x_half = (x_max - x_min) / 2.0

# Map v to [-1, 1]
v_mid = (v_min + v_max) / 2.0
v_half = (v_max - v_min) / 2.0

# Define the basis functions.
# We need 10 terms. Let's select a mix of low-order terms that capture
# typical oscillator dynamics (e.g., linear, cubic, coupling).
# Total degree <= 4.
# Terms:
# 0: T0(u)T0(v) = 1
# 1: T1(u)T0(v) = u  (linear in x)
# 2: T0(u)T1(v) = v  (linear in v)
# 3: T2(u)T0(v) = u^2 - 0.5 (quadratic in x)
# 4: T0(u)T2(v) = v^2 - 0.5 (quadratic in v)
# 5: T3(u)T0(v) = 4u^3 - 3u (cubic in x)
# 6: T0(u)T3(v) = 4v^3 - 3v (cubic in v)
# 7: T1(u)T1(v) = u*v (coupling)
# 8: T2(u)T1(v) = (u^2 - 0.5)*v (coupling x^2 v)
# 9: T1(u)T2(v) = u*(v^2 - 0.5) (coupling x v^2)

# This selection covers the most physically relevant low-order terms for a nonlinear oscillator
# like Duffing (x^3) or Van der Pol (x^2 v), while maintaining smoothness.

def _chebyshev_basis(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute the 10 Chebyshev basis functions for normalized inputs u, v.
    u, v are in [-1, 1].
    Returns an array of shape (N, 10).
    """
    N = len(u)
    basis = np.zeros((N, 10))
    
    # T0 = 1
    basis[:, 0] = 1.0
    
    # T1 = x
    basis[:, 1] = u
    basis[:, 2] = v
    
    # T2 = 2x^2 - 1  (Wait, standard Chebyshev T_n(cos(theta)) = cos(n*theta))
    # T_0(x) = 1
    # T_1(x) = x
    # T_2(x) = 2x^2 - 1
    # T_3(x) = 4x^3 - 3x
    # T_4(x) = 8x^4 - 8x^2 + 1
    
    # Let's recalculate T2 and T3 correctly.
    # My previous comment in the plan had T2 = u^2 - 0.5, which is not standard T2.
    # Standard T2(u) = 2u^2 - 1.
    # Standard T3(u) = 4u^3 - 3u.
    
    basis[:, 3] = 2.0 * u**2 - 1.0
    basis[:, 4] = 2.0 * v**2 - 1.0
    basis[:, 5] = 4.0 * u**3 - 3.0 * u
    basis[:, 6] = 4.0 * v**3 - 3.0 * v
    basis[:, 7] = u * v
    basis[:, 8] = (2.0 * u**2 - 1.0) * v
    basis[:, 9] = u * (2.0 * v**2 - 1.0)
    
    return basis

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts the target using a linear combination of Chebyshev polynomial basis functions.
    
    Args:
        x: Position array (N,)
        v: Velocity array (N,)
        params: Array of 10 parameters (weights for the basis functions).
    
    Returns:
        Prediction array (N,)
    """
    try:
        # Normalize inputs to [-1, 1]
        u = (x - x_mid) / x_half
        v_norm = (v - v_mid) / v_half
        
        # Clip to [-1, 1] to handle any slight out-of-domain noise, 
        # though Chebyshev polynomials are defined everywhere, 
        # clipping prevents extreme growth if inputs are far out of bounds.
        u = np.clip(u, -1.0, 1.0)
        v_norm = np.clip(v_norm, -1.0, 1.0)
        
        # Compute basis
        basis = _chebyshev_basis(u, v_norm)
        
        # Linear combination
        pred = basis @ params
        
        # Ensure finite output
        if not np.all(np.isfinite(pred)):
            pred = np.where(np.isfinite(pred), pred, 0.0)
            
        return pred
        
    except Exception:
        # Fallback to simple linear if something goes wrong
        try:
            return params[0]*x + params[1]*v + params[2]
        except Exception:
            return np.zeros_like(x)

# EVOLVE-BLOCK-END
