import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Global rational function model using Chebyshev polynomials.
    f(x) = (sum_{i=0}^{4} a_i T_i(x)) / (1 + sum_{j=1}^{4} b_j T_j(x))
    
    This model can capture singularities and asymptotic behaviors.
    The denominator is regularized to avoid division by zero.
    
    Params:
    0-4: a_0, a_1, a_2, a_3, a_4 (numerator coefficients)
    5-9: b_1, b_2, b_3, b_4, b_5 (denominator coefficients, b_0 fixed to 1)
    """
    a0, a1, a2, a3, a4 = params[0], params[1], params[2], params[3], params[4]
    b1, b2, b3, b4, b5 = params[5], params[6], params[7], params[8], params[9]
    
    # Compute Chebyshev polynomials T_0 through T_5
    # T_0(x) = 1
    # T_1(x) = x
    # T_2(x) = 2x^2 - 1
    # T_3(x) = 4x^3 - 3x
    # T_4(x) = 8x^4 - 8x^2 + 1
    # T_5(x) = 16x^5 - 20x^3 + 5x
    
    T0 = np.ones_like(x)
    T1 = x
    T2 = 2 * x**2 - 1
    T3 = 4 * x**3 - 3 * x
    T4 = 8 * x**4 - 8 * x**2 + 1
    T5 = 16 * x**5 - 20 * x**3 + 5 * x
    
    # Numerator: sum_{i=0}^{4} a_i T_i(x)
    numerator = a0 * T0 + a1 * T1 + a2 * T2 + a3 * T3 + a4 * T4
    
    # Denominator: 1 + sum_{j=1}^{5} b_j T_j(x)
    # Note: We use T_1 to T_5 for the denominator coefficients b_1 to b_5
    # This gives us 5 free parameters for the denominator plus the fixed 1.
    denominator = 1.0 + b1 * T1 + b2 * T2 + b3 * T3 + b4 * T4 + b5 * T5
    
    # Regularize denominator to avoid division by zero
    # Add a small epsilon to prevent exact zeros
    eps = 1e-8
    # Use np.where to handle sign and magnitude
    # If denominator is close to 0, replace with eps * sign(denominator) or just eps
    # To be safe, we can clip the denominator away from 0
    # However, BFGS might struggle if we introduce non-differentiable points.
    # Instead, we can use a smooth approximation or just add a large constant if needed.
    # But the task says "well-scaled rational". 
    # Let's try a simple safeguard: if |denom| < eps, use eps.
    # This is non-smooth at 0. 
    # Alternative: denominator = 1.0 + b1*T1 + ... + b5*T5 + 1e-4
    # This shifts the pole away from the real axis slightly or ensures denom != 0.
    # Adding a constant to the denominator changes the function.
    # Better: ensure the denominator stays away from 0 by construction?
    # Or just rely on the optimizer not hitting exact zeros, and add a tiny epsilon in division.
    
    # Let's use a safe division
    safe_denom = np.where(np.abs(denominator) < eps, eps * np.sign(denominator), denominator)
    # np.sign(0) is 0, so if denom is exactly 0, sign is 0, safe_denom is 0. Bad.
    # Use: np.where(np.abs(denominator) < eps, eps, denominator) ? No, loses sign.
    # Use: np.where(denominator >= 0, np.maximum(denominator, eps), np.minimum(denominator, -eps))
    
    safe_denom = np.where(denominator >= 0, np.maximum(denominator, eps), np.minimum(denominator, -eps))
    
    y = numerator / safe_denom
    
    return y
