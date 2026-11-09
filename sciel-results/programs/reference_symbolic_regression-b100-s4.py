import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # Strategy: Linear Basis Function Regression
    # We map the 10 parameters to a fixed set of global basis functions.
    # This makes the optimization problem convex (minimizing MSE of linear combination),
    # allowing BFGS to converge reliably to the global optimum for the given basis,
    # avoiding the local minima traps of non-linear basis parameters (like Gaussian centers/widths).
    
    # Basis functions chosen to cover a wide range of potential shapes on [-2, 2]:
    # p[0]: Constant (1)
    # p[1]: Linear (x)
    # p[2]: Quadratic (x^2)
    # p[3]: Cubic (x^3)
    # p[4]: Quartic (x^4)
    # p[5]: Quintic (x^5)
    # p[6]: Sextic (x^6)
    # p[7]: Septic (x^7)
    # p[8]: Sin(x)
    # p[9]: Cos(x)
    
    # Note: Higher order polynomials can be ill-conditioned, but BFGS with default 
    # tolerances usually handles this well for small N=10. 
    # We could use Chebyshev polynomials for better conditioning, but standard powers 
    # are often sufficient if the target isn't extremely high-frequency.
    # Let's stick to standard powers and trig for simplicity and broad coverage.
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x**2
    b3 = x**3
    b4 = x**4
    b5 = x**5
    b6 = x**6
    b7 = x**7
    b8 = np.sin(x)
    b9 = np.cos(x)
    
    y = (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 + p[4] * b4 + 
         p[5] * b5 + p[6] * b6 + p[7] * b7 + p[8] * b8 + p[9] * b9)
         
    return y
