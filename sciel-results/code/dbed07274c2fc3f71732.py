"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Separable Fourier series model:
    sin(a*x)*sin(b*y) + sin(c*x)*sin(d*y) + e*x*y + f
    
    params[0] = a (frequency for first sin in x)
    params[1] = b (frequency for first sin in y)
    params[2] = c (frequency for second sin in x)
    params[3] = d (frequency for second sin in y)
    params[4] = e (amplitude for x*y term)
    params[5] = f (constant offset)
    params[6] = amplitude for first product term
    params[7] = amplitude for second product term
    params[8] = additional frequency for third term (cos)
    params[9] = amplitude for third term
    
    Model:
    pred = p6 * sin(p0*x)*sin(p1*y) + p7 * sin(p2*x)*sin(p3*y) 
         + p4*x*y + p5 + p9 * cos(p8*x)*cos(p8*y)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Check if params are valid
    if not np.all(np.isfinite(params)):
        return np.zeros(len(x))
        
    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]
    e = params[4]
    f = params[5]
    amp1 = params[6]
    amp2 = params[7]
    freq3 = params[8]
    amp3 = params[9]
    
    # Compute the separable Fourier terms
    # Term 1: sin(a*x) * sin(b*y)
    term1 = np.sin(a * x) * np.sin(b * y)
    
    # Term 2: sin(c*x) * sin(d*y)
    term2 = np.sin(c * x) * np.sin(d * y)
    
    # Term 3: x*y (bilinear)
    term3 = x * y
    
    # Term 4: cos(freq3*x) * cos(freq3*y)
    term4 = np.cos(freq3 * x) * np.cos(freq3 * y)
    
    # Combine
    pred = amp1 * term1 + amp2 * term2 + e * term3 + f + amp3 * term4
    
    # Ensure finite
    if not np.all(np.isfinite(pred)):
        return np.zeros(len(x))
        
    return pred
# EVOLVE-BLOCK-END
