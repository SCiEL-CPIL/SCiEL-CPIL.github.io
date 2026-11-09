import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Model: y = sum_{i=0}^{4} A_i * exp(-((x - c_i) / s_i)^2)
    # Params layout:
    # p[0..4]  : Amplitudes A_i
    # p[5..9]  : Centers c_i
    # We fix the widths s_i to a reasonable value to keep the optimization landscape smooth
    # and avoid singularities when s_i -> 0. 
    # Alternatively, we could use p[5..9] as widths, but centers are usually more critical 
    # for location. Let's use a hybrid: 
    # Actually, with only 10 params, fitting 5 centers and 5 amplitudes is a good trade-off.
    # We will fix the width to 0.5, which covers about 1/4 of the domain [-2, 2] per bump.
    # This allows the model to represent functions like sin(x), cos(x), polynomials (roughly), 
    # or piecewise features.
    
    A = p[0:5]
    c = p[5:10]
    
    # Fixed widths to ensure numerical stability and prevent BFGS from getting stuck
    # in narrow valleys where widths approach 0.
    widths = np.full(5, 0.5, dtype=np.float64)
    
    # Compute prediction efficiently
    # x shape: (N,)
    # A, c, widths shape: (5,)
    # We need (N, 5) matrix of Gaussian values
    
    # Check time budget
    if time.time() - start_time > deadline:
        return np.zeros_like(x)
        
    # Reshape for broadcasting: x[:, None] is (N, 1), c[None, :] is (1, 5)
    diff = x[:, None] - c[None, :]
    
    # Compute squared distance
    diff_sq = diff * diff
    
    # Divide by width squared
    width_sq = widths[None, :] * widths[None, :]
    normalized_sq = diff_sq / width_sq
    
    # Compute exponent
    # Clip to avoid underflow/overflow issues, though exp(-large) is just 0
    # exp(-1000) is fine in float64, but let's be safe
    exponents = -normalized_sq
    
    # Compute Gaussians
    gaussians = np.exp(exponents)
    
    # Weighted sum
    y_pred = gaussians @ A
    
    return y_pred
