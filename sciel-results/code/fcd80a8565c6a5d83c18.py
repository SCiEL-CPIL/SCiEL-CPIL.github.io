import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # Strategy: Product of Two Low-Degree Polynomials
    # We model y(x) = P_5(x) * Q_4(x)
    # where P_5(x) = p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4 + p5*x^5
    # and Q_4(x) = p6 + p7*x + p8*x^2 + p9*x^3
    # Note: Q_4 is actually degree 3 here to keep total params at 10 (6 for P, 4 for Q).
    # Let's adjust: P_5 (6 params) and Q_3 (4 params). Total degree 8.
    # This creates a non-convex optimization problem for BFGS.
    
    # Split parameters
    p_p = p[0:6]  # Coefficients for P(x)
    p_q = p[6:10] # Coefficients for Q(x)
    
    # Compute P(x)
    # P(x) = p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4 + p5*x^5
    # Using Horner's method for stability and speed
    P = p_p[0]
    for i in range(1, 6):
        P = P * x + p_p[i]
        
    # Compute Q(x)
    # Q(x) = p6 + p7*x + p8*x^2 + p9*x^3
    Q = p_q[0]
    for i in range(1, 4):
        Q = Q * x + p_q[i]
        
    y = P * Q
    
    return y
