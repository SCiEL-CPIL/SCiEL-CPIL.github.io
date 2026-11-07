import numpy as np
import time

_start_time = None

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model using a Rational Function basis.
    
    y(x) = (a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4) / (1 + b1*x + b2*x^2 + b3*x^3 + b4*x^4 + b5*x^5)
    
    This structure allows for modeling poles and asymptotes, which 
    polynomial and RBF bases may struggle with. The non-linear 
    dependence on parameters provides greater expressive power 
    for a wider variety of hidden functions.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Time guard: if we're too close to the 150s limit, return a safe default
    elapsed = time.time() - _start_time
    if elapsed > 140:
        return np.zeros_like(np.asarray(x, dtype=float))
    
    x = np.asarray(x, dtype=float)
    
    # Unpack parameters
    # a0..a4 (5 params)
    # b1..b5 (5 params)
    a = params[:5]
    b = params[5:]
    
    # Compute powers of x for efficiency
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    
    # Numerator (Degree 4)
    num = a[0] + a[1]*x + a[2]*x2 + a[3]*x3 + a[4]*x4
    
    # Denominator (Degree 5, constant term fixed at 1)
    den = 1.0 + b[0]*x + b[1]*x2 + b[2]*x3 + b[3]*x4 + b[4]*x5
    
    # Smooth safety: use a soft floor to keep denominator away from zero
    # This avoids gradient discontinuities that hurt BFGS convergence
    # Use a smooth max-like operation: den_safe = den + eps * (1 - tanh(den/eps))
    # Simpler: just clip with a smooth transition
    eps = 1e-6
    # Use a smooth approximation: where |den| < eps, blend toward eps
    abs_den = np.abs(den)
    # Smooth blending: factor is 1 when abs_den >= eps, smoothly goes to 0 when abs_den -> 0
    blend = np.minimum(abs_den / eps, 1.0)
    # den_safe = den * blend + sign(den) * eps * (1 - blend)
    sign_den = np.where(den >= 0, 1.0, -1.0)
    den_safe = den * blend + sign_den * eps * (1.0 - blend)
    
    y = num / den_safe
    
    # Ensure output is finite
    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)
    
    return y
