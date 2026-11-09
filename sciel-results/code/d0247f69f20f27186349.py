import numpy as np
import time

# Precompute Chebyshev nodes and basis functions on the normalized domain
# Domain: x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
x_min, x_max = -0.5728, 0.5275
v_min, v_max = -0.1754, 0.2821

x_center = (x_max + x_min) / 2.0
x_scale = (x_max - x_min) / 2.0
v_center = (v_max + v_min) / 2.0
v_scale = (v_max - v_min) / 2.0

# We have 10 parameters.
# Let's use a tensor product of Chebyshev polynomials.
# Degrees: (0,0), (1,0), (0,1), (2,0), (0,2), (1,1), (3,0), (0,3), (2,1), (1,2)
# This covers up to degree 3 in x and 3 in v, and cross terms up to degree 2.
# Total 10 terms.
# This is a structured polynomial basis that is better conditioned than monomials.
# Alternatively, we can use a mix.

# Let's define the basis functions explicitly.
# T0(t) = 1
# T1(t) = t
# T2(t) = 2t^2 - 1
# T3(t) = 4t^3 - 3t

def chebyshev_t(t, n):
    if n == 0:
        return np.ones_like(t)
    elif n == 1:
        return t
    elif n == 2:
        return 2 * t**2 - 1
    elif n == 3:
        return 4 * t**3 - 3 * t
    else:
        # Recursive, but we only need up to 3
        pass

# Pre-compute nothing, compute on the fly as it's cheap.
# However, to ensure speed and safety, we can just write the expression directly.

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Structured Chebyshev polynomial expansion for a nonlinear oscillator.
    Inputs x and v are normalized to [-1, 1] approximately.
    params is an array of 10 coefficients.
    """
    # Normalize inputs to [-1, 1]
    xn = (x - x_center) / x_scale
    vn = (v - v_center) / v_scale
    
    # Compute Chebyshev basis functions
    # T0 = 1
    # T1 = t
    # T2 = 2t^2 - 1
    # T3 = 4t^3 - 3t
    
    # x-basis
    x0 = np.ones_like(xn)
    x1 = xn
    x2 = 2 * xn**2 - 1
    x3 = 4 * xn**3 - 3 * xn
    
    # v-basis
    v0 = np.ones_like(vn)
    v1 = vn
    v2 = 2 * vn**2 - 1
    v3 = 4 * vn**3 - 3 * vn
    
    # Construct the 10 terms
    # 0: T0(x)T0(v)
    # 1: T1(x)T0(v)
    # 2: T0(x)T1(v)
    # 3: T2(x)T0(v)
    # 4: T0(x)T2(v)
    # 5: T1(x)T1(v)
    # 6: T3(x)T0(v)
    # 7: T0(x)T3(v)
    # 8: T2(x)T1(v)
    # 9: T1(x)T2(v)
    
    term0 = x0 * v0
    term1 = x1 * v0
    term2 = x0 * v1
    term3 = x2 * v0
    term4 = x0 * v2
    term5 = x1 * v1
    term6 = x3 * v0
    term7 = x0 * v3
    term8 = x2 * v1
    term9 = x1 * v2
    
    # Combine with parameters
    result = (params[0] * term0 + 
              params[1] * term1 + 
              params[2] * term2 + 
              params[3] * term3 + 
              params[4] * term4 + 
              params[5] * term5 + 
              params[6] * term6 + 
              params[7] * term7 + 
              params[8] * term8 + 
              params[9] * term9)
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
