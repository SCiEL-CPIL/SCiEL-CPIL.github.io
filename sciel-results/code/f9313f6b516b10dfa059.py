import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # Strategy: Linear Basis Function Regression with Chebyshev and Trig Basis
    # Chebyshev polynomials are better conditioned than monomials on [-1, 1].
    # We map x in [-2, 2] to t in [-1, 1] via t = 0.5 * x.
    # We use T0..T5 for polynomial trends and sin/cos for periodic trends.
    
    t = 0.5 * x
    
    # Chebyshev Polynomials of the First Kind
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    T5 = 16.0 * t**5 - 20.0 * t**3 + 5.0 * t
    
    # Trigonometric Functions
    S1 = np.sin(x)
    C1 = np.cos(x)
    S2 = np.sin(2.0 * x)
    C2 = np.cos(2.0 * x)
    
    y = (p[0] * T0 + p[1] * T1 + p[2] * T2 + p[3] * T3 + p[4] * T4 + 
         p[5] * T5 + p[6] * S1 + p[7] * C1 + p[8] * S2 + p[9] * C2)
         
    return y
