import numpy as np
import time

# Deadline guard: 150s wall-clock limit, use 145s to be safe
_DEADLINE = 145.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with 1 input.
    
    Strategy: Use a flexible basis that combines:
    1. Fourier-like terms (sine/cosine with various frequencies)
    2. Polynomial terms
    3. Gaussian RBFs with fixed centers and widths
    
    The model is linear in params, so BFGS will find the global optimum efficiently.
    
    With 10 params, we can have a rich basis. Let's use:
    - 2 polynomial terms: 1, x
    - 4 trigonometric terms: sin(x), cos(x), sin(2x), cos(2x)
    - 4 Gaussian RBFs with centers at -1.5, -0.5, 0.5, 1.5 and width 0.5
    """
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Ensure params is a flat array of length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
    
    try:
        # Build basis functions
        # Polynomial: [1, x]
        # Trig: [sin(x), cos(x), sin(2x), cos(2x)]
        # RBF: [exp(-(x-c)^2/(2*0.5^2)) for c in [-1.5, -0.5, 0.5, 1.5]]
        
        n = x.size
        
        # Polynomial basis
        p0 = np.ones_like(x)
        p1 = x.copy()
        
        # Trig basis
        t0 = np.sin(x)
        t1 = np.cos(x)
        t2 = np.sin(2*x)
        t3 = np.cos(2*x)
        
        # RBF basis
        centers = np.array([-1.5, -0.5, 0.5, 1.5])
        width = 0.5
        sig2 = 2.0 * width**2
        r0 = np.exp(-(x - centers[0])**2 / sig2)
        r1 = np.exp(-(x - centers[1])**2 / sig2)
        r2 = np.exp(-(x - centers[2])**2 / sig2)
        r3 = np.exp(-(x - centers[3])**2 / sig2)
        
        # Stack into basis matrix (n x 10)
        basis = np.column_stack([p0, p1, t0, t1, t2, t3, r0, r1, r2, r3])
        
        # Linear model: y = basis @ params
        return basis @ params
        
    except Exception:
        return np.zeros_like(x)
