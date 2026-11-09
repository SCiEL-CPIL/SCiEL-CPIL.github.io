import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational function model using Chebyshev polynomials.
    
    f(x) = N(t) / D(t)
    where t = x/2 maps [-2, 2] to [-1, 1]
    N(t) = sum_{i=0}^{4} a_i T_i(t)
    D(t) = sum_{j=0}^{4} b_j T_j(t)
    
    This allows the model to capture singularities, poles, and asymptotic
    behaviors that polynomial or radial-basis models cannot represent well.
    The Chebyshev basis provides numerical stability on [-1, 1].
    
    Parameters (10 total):
    params[0..4]  = a0..a4 (numerator coefficients)
    params[5..9]  = b0..b4 (denominator coefficients)
    """
    # Unpack parameters
    a0, a1, a2, a3, a4 = params[0], params[1], params[2], params[3], params[4]
    b0, b1, b2, b3, b4 = params[5], params[6], params[7], params[8], params[9]
    
    # Map x from [-2, 2] to t in [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 through T_4
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2 * t * t - 1
    T3 = 4 * t**3 - 3 * t
    T4 = 8 * t**4 - 8 * t**2 + 1
    
    # Numerator: N(t) = a0*T0 + a1*T1 + a2*T2 + a3*T3 + a4*T4
    N = a0 * T0 + a1 * T1 + a2 * T2 + a3 * T3 + a4 * T4
    
    # Denominator: D(t) = b0*T0 + b1*T1 + b2*T2 + b3*T3 + b4*T4
    D = b0 * T0 + b1 * T1 + b2 * T2 + b3 * T3 + b4 * T4
    
    # Compute ratio with safety against division by zero
    # Use a small epsilon to avoid division by zero
    eps = 1e-10
    D_safe = D + np.sign(D) * eps
    D_safe = np.where(np.abs(D) < eps, eps, D_safe)
    
    result = N / D_safe
    
    # Ensure finite outputs
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
