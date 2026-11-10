import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function.
    Adapts model complexity based on the number of parameters provided.
    """
    x = np.asarray(x, dtype=np.float64)
    n_params = len(params)
    
    # Compute powers of x up to degree 9 (max needed for 10 params model)
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    x6 = x5 * x
    x7 = x6 * x
    x8 = x7 * x
    x9 = x8 * x
    
    powers = [np.ones_like(x), x, x2, x3, x4, x5, x6, x7, x8, x9]
    
    if n_params >= 10:
        # Original strategy: P5(x) * Q4(x)
        a = params[0:6]
        b = params[6:10]
        
        p_val = a[0]*powers[0] + a[1]*powers[1] + a[2]*powers[2] + a[3]*powers[3] + a[4]*powers[4] + a[5]*powers[5]
        q_val = b[0]*powers[0] + b[1]*powers[1] + b[2]*powers[2] + b[3]*powers[3] + b[4]*powers[4]
        
        return p_val * q_val
    elif n_params == 4:
        # Fallback for size 4: Cubic polynomial
        c = params[0:4]
        return c[0]*powers[0] + c[1]*powers[1] + c[2]*powers[2] + c[3]*powers[3]
    elif n_params == 5:
        # Fallback for size 5: Quartic polynomial
        c = params[0:5]
        return c[0]*powers[0] + c[1]*powers[1] + c[2]*powers[2] + c[3]*powers[3] + c[4]*powers[4]
    else:
        # Generic fallback: Use available params as coefficients for a polynomial
        # This ensures we don't crash even if size is 1, 2, 3, 6, 7, 8, 9
        result = np.zeros_like(x)
        for i in range(n_params):
            result += params[i] * powers[i]
        return result
