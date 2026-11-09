import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure x is 1D
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
    
    # Extract parameters
    p = np.asarray(params, dtype=np.float64)
    
    # Check for non-finite parameters
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # Structure:
    # p[0]: Global bias
    # p[1]: Amplitude 1
    # p[2]: Center 1
    # p[3]: Log-width 1
    # p[4]: Amplitude 2
    # p[5]: Center 2
    # p[6]: Log-width 2
    # p[7]: Amplitude 3
    # p[8]: Center 3
    # p[9]: Log-width 3
    
    # Initialize output with bias
    y = np.full_like(x, p[0], dtype=np.float64)
    
    # Define centers with fixed offsets to break symmetry at initialization (all ones)
    # Centers: p[2], p[5]+1.0, p[8]-1.0
    # Widths: exp(p[3]), exp(p[6]), exp(p[9])
    
    # Gaussian 1
    c1 = p[2]
    w1 = np.exp(p[3])
    # Avoid division by zero if width is extremely small
    if w1 < 1e-8:
        w1 = 1e-8
    y += p[1] * np.exp(-0.5 * ((x - c1) / w1) ** 2)
    
    # Gaussian 2
    c2 = p[5] + 1.0
    w2 = np.exp(p[6])
    if w2 < 1e-8:
        w2 = 1e-8
    y += p[4] * np.exp(-0.5 * ((x - c2) / w2) ** 2)
    
    # Gaussian 3
    c3 = p[8] - 1.0
    w3 = np.exp(p[9])
    if w3 < 1e-8:
        w3 = 1e-8
    y += p[7] * np.exp(-0.5 * ((x - c3) / w3) ** 2)
    
    # Final check for non-finite values
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
        
    return y
