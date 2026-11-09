import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Softplus-based nonlinear oscillator model.
    
    Uses a smooth, differentiable softplus-based basis for nonlinearities
    in position and velocity, with periodic driving terms.
    
    Features:
    - Position: x, softplus(x), softplus(-x), x*softplus(x), x*softplus(-x)
    - Velocity: v, softplus(v), softplus(-v), v*softplus(v), v*softplus(-v)
    - Time: cos(t), sin(t)
    - Cross: x*v
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Softplus: log(1 + exp(x)) - smooth approximation of max(0, x)
    # Use stable computation to avoid overflow
    def softplus(z):
        return np.log1p(np.exp(np.clip(z, -50, 50)))
    
    sp_x = softplus(x)
    sp_neg_x = softplus(-x)
    sp_v = softplus(v)
    sp_neg_v = softplus(-v)
    
    # Build feature vector
    # Position features
    f_x1 = x
    f_x2 = sp_x
    f_x3 = sp_neg_x
    f_x4 = x * sp_x
    f_x5 = x * sp_neg_x
    
    # Velocity features
    f_v1 = v
    f_v2 = sp_v
    f_v3 = sp_neg_v
    f_v4 = v * sp_v
    f_v5 = v * sp_neg_v
    
    # Time features
    f_t1 = np.cos(t)
    f_t2 = np.sin(t)
    
    # Cross feature
    f_xv = x * v
    
    # Combine with parameters
    # Use all 10 params distributed across features
    # We have 12 features but only 10 params, so we combine some
    # p0: offset
    # p1: x
    # p2: sp_x - sp_neg_x (antisymmetric position nonlinearity)
    # p3: x * (sp_x + sp_neg_x) (symmetric position nonlinearity)
    # p4: v
    # p5: sp_v - sp_neg_v (antisymmetric velocity nonlinearity)
    # p6: v * (sp_v + sp_neg_v) (symmetric velocity nonlinearity)
    # p7: cos(t)
    # p8: sin(t)
    # p9: x * v
    
    a = (p0 
         + p1 * f_x1
         + p2 * (f_x2 - f_x3)
         + p3 * f_x1 * (f_x2 + f_x3)
         + p4 * f_v1
         + p5 * (f_v2 - f_v3)
         + p6 * f_v1 * (f_v2 + f_v3)
         + p7 * f_t1
         + p8 * f_t2
         + p9 * f_xv)
    
    return a
