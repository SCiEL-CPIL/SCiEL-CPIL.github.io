import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Scaled polynomial interaction model for nonlinear oscillator.
    
    This model uses a basis of polynomial terms in normalized time, position, and velocity.
    Parameters are scaled by the input ranges to ensure balanced gradients for BFGS optimization.
    
    Basis:
    1. Bias
    2. x
    3. v
    4. t
    5. x^2
    6. v^2
    7. x*v
    8. t*x
    9. t*v
    10. x^3
    """
    # Input ranges from task description
    # t: [30, 50]
    # x: [-0.2056, 0.1749]
    # v: [-0.3391, 0.3624]
    
    t_min, t_max = 30.0, 50.0
    x_min, x_max = -0.2056, 0.1749
    v_min, v_max = -0.3391, 0.3624
    
    # Centers and scales (half-ranges)
    t_c = (t_min + t_max) / 2.0
    x_c = (x_min + x_max) / 2.0
    v_c = (v_min + v_max) / 2.0
    
    t_s = (t_max - t_min) / 2.0
    x_s = (x_max - x_min) / 2.0
    v_s = (v_max - v_min) / 2.0
    
    # Normalize inputs to approx [-1, 1]
    ts = (t - t_c) / t_s
    xs = (x - x_c) / x_s
    vs = (v - v_c) / v_s
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale parameters to match the magnitude of the unnormalized terms
    # This is crucial for BFGS starting from ones.
    # If we use normalized inputs (xs, vs, ts), the coefficients should be O(1) 
    # relative to the target variance. 
    # However, the target y is in [-0.71, 0.66], var ~ 0.1-0.2.
    # The normalized inputs are O(1). So coefficients p0..p9 should be O(1) or smaller.
    # Starting from 1.0 is a reasonable guess for O(1) coefficients.
    
    # Compute terms
    term0 = p0
    term1 = p1 * xs
    term2 = p2 * vs
    term3 = p3 * ts
    term4 = p4 * (xs * xs)
    term5 = p5 * (vs * vs)
    term6 = p6 * (xs * vs)
    term7 = p7 * (ts * xs)
    term8 = p8 * (ts * vs)
    term9 = p9 * (xs * xs * xs)
    
    result = term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
    
    return result
