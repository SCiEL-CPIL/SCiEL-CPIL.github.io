import time
import numpy as np
from scipy.optimize import minimize

N = 600
TOTAL_TIME = 150.0
SAFETY_MARGIN = 5.0

def _autoconv_peak(f):
    """Compute the normalized maximum of the autoconvolution of f."""
    dx = 1.0 / (2.0 * N)
    size = 2 * N
    f_padded = np.zeros(size)
    f_padded[:N] = f
    
    F = np.fft.fft(f_padded, n=2*size)
    conv = np.fft.ifft(F * F).real[:size]
    
    peak = np.max(conv) * dx
    l1 = np.sum(f) * dx
    normalized_peak = peak / (l1 * l1)
    
    return normalized_peak

def construct():
    start_time = time.time()
    deadline = start_time + TOTAL_TIME - SAFETY_MARGIN
    
    # Initialize with a good heuristic
    # The uniform function is a critical point but not the global min.
    # A "triangle" or "parabola" shape often works better.
    t = np.linspace(-0.25, 0.25, N, endpoint=False)
    # Try a few starts
    candidates = [
        np.ones(N),
        np.cos(np.pi * t) ** 2, # Cosine bump
        np.exp(-10 * t**2), # Gaussian
        np.maximum(0, 1 - 4*np.abs(t)) # Triangle
    ]
    
    best_f = None
    best_val = float('inf')
    
    for c in candidates:
        c = np.maximum(0, c)
        val = _autoconv_peak(c)
        if val < best_val:
            best_val = val
            best_f = c.copy()
            
    f = best_f.copy()
    current_val = best_val
    
    # Block Coordinate Descent
    B = 10 # Block size
    
    # Pre-allocate some arrays to reduce overhead?
    # Not strictly necessary, but helps.
    
    while time.time() < deadline:
        # Pick a random block
        start_idx = np.random.randint(0, N - B)
        idxs = np.arange(start_idx, start_idx + B)
        
        # Current values for this block
        x0 = f[idxs].copy()
        
        # Define local objective
        def local_obj(x):
            f_new = f.copy()
            f_new[idxs] = x
            return _autoconv_peak(f_new)
            
        # Optimize this block
        # Use L-BFGS-B for bounded optimization
        bounds = [(0, None)] * B
        
        try:
            res = minimize(
                local_obj, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 20, 'ftol': 1e-5, 'gtol': 1e-5}
            )
            if res.success or res.fun < current_val:
                f[idxs] = res.x
                new_val = res.fun
                if new_val < current_val:
                    current_val = new_val
                    if new_val < best_val:
                        best_val = new_val
                        best_f = f.copy()
        except Exception:
            pass
            
    return list(best_f)

if __name__ == "__main__":
    result = construct()
    print(f"Result: {len(result)} values, sum={sum(result):.4f}")
