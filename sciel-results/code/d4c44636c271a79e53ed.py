import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Learnable Frequency Fourier Series with Polynomial and Exponential Terms.
    
    This strategy uses a hybrid basis where the frequencies of the sinusoidal
    terms are themselves parameters. This allows the model to adapt to the
    specific periodicity of the target function, unlike fixed-frequency Fourier
    series. It also retains polynomial and exponential terms to capture
    non-periodic trends.
    
    Parameters (10 total):
    p[0]: coeff for 1
    p[1]: coeff for x
    p[2]: coeff for x^2
    p[3]: coeff for exp(-x)
    p[4]: coeff for exp(x)
    p[5]: freq for sin term 1 (f1)
    p[6]: coeff for sin(f1 * x)
    p[7]: freq for cos term 1 (f2)
    p[8]: coeff for cos(f2 * x)
    p[9]: coeff for x^3
    
    Note: The frequencies p[5] and p[7] are scaled to keep them in a reasonable
    range. BFGS will optimize these.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    p = np.asarray(params, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] != 10:
        return np.zeros_like(x)
    
    c0, c1, c2, c3, c4, f1, c5, f2, c6, c7 = p
    
    # Compute basis functions
    # Polynomials
    b0 = np.ones_like(x)
    b1 = x
    b2 = x ** 2
    b9 = x ** 3
    
    # Exponentials
    # Clip x to avoid overflow in exp(x) if x gets large, though domain is [0.1, 6]
    x_clip = np.clip(x, 0, 10)
    b3 = np.exp(-x_clip)
    b4 = np.exp(x_clip)
    
    # Learnable Frequencies
    # We scale the frequency parameters to be positive and reasonable.
    # Using a softplus-like transformation or just clipping to [0, 20]
    f1_scaled = np.clip(f1, 0.1, 20.0)
    f2_scaled = np.clip(f2, 0.1, 20.0)
    
    # Fourier terms with learned frequencies
    b5 = np.sin(f1_scaled * x)
    b6 = np.cos(f2_scaled * x)
    
    # Linear combination
    result = (c0 * b0 + 
              c1 * b1 + 
              c2 * b2 + 
              c3 * b3 + 
              c4 * b4 + 
              c5 * b5 + 
              c6 * b6 + 
              c7 * b9)
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
