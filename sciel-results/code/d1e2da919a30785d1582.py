import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
    
    # Rational function model: y(x) = P_4(x) / Q_4(x)
    # P_4 and Q_4 are degree-4 polynomials with coefficients from params
    # params[0:5] -> numerator coefficients (constant to x^4)
    # params[5:10] -> denominator coefficients (constant to x^4)
    
    # Ensure x is 2D for consistent output shape
    x_1d = x.flatten()
    x_2d = x_1d[:, np.newaxis]
    
    # Compute powers of x: [1, x, x^2, x^3, x^4]
    x_powers = np.hstack([
        np.ones_like(x_1d)[:, np.newaxis],
        x_2d,
        x_2d ** 2,
        x_2d ** 3,
        x_2d ** 4
    ])  # shape: (N, 5)
    
    # Numerator: P_4(x) = sum_{i=0}^{4} p_i * x^i
    p_coeffs = params[:5]
    numerator = x_powers @ p_coeffs  # shape: (N,)
    
    # Denominator: Q_4(x) = sum_{i=0}^{4} q_i * x^i
    q_coeffs = params[5:10]
    denominator = x_powers @ q_coeffs  # shape: (N,)
    
    # Handle potential division by zero or near-zero denominators
    # Use a small epsilon to avoid numerical instability
    eps = 1e-10
    safe_denominator = np.where(np.abs(denominator) < eps, eps * np.sign(denominator + 1e-30), denominator)
    
    result = numerator / safe_denominator
    
    # Reshape to match input shape
    result = result.reshape(x.shape)
    
    return result
