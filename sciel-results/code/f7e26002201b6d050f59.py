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
    
    # Parametric representation: f(t) = sum_{k=0}^{K-1} a_k * cos(2 * pi * k * t)
    # This reduces the search space from 600 dimensions to K dimensions.
    # We use Nelder-Mead (derivative-free) to handle the non-smoothness introduced by clipping.
    
    K = 15 # Number of cosine terms
    t = np.linspace(-0.25, 0.25, N, endpoint=False)
    
    # Pre-compute basis matrix
    basis = np.zeros((N, K))
    for k in range(K):
        basis[:, k] = np.cos(2 * np.pi * k * t)
        
    def param_obj(params):
        # params: [a_0, a_1, ..., a_{K-1}]
        f_vals = basis @ params
        # Enforce non-negativity
        f_vals = np.maximum(0, f_vals)
        if np.sum(f_vals) < 1e-10:
            return 1e6
        return _autoconv_peak(f_vals)
    
    best_f = None
    best_val = float('inf')
    
    # Multi-start Nelder-Mead
    np.random.seed(42)
    
    n_starts = 30
    start_points = []
    
    # Deterministic starts
    p0 = np.zeros(K)
    p0[0] = 1.0
    start_points.append(p0)
    
    p1 = np.zeros(K)
    p1[0] = 1.0
    p1[1] = -0.5
    start_points.append(p1)
    
    p2 = np.zeros(K)
    p2[0] = 0.5
    p2[1] = 0.5
    start_points.append(p2)
    
    p3 = np.zeros(K)
    p3[0] = 2.0
    p3[1] = -1.0
    p3[2] = 0.5
    start_points.append(p3)
    
    # Random starts
    for _ in range(n_starts - 4):
        p_rand = np.random.randn(K) * 0.5
        p_rand[0] = np.abs(p_rand[0]) + 0.5 # Ensure some mass
        start_points.append(p_rand)
        
    for i, p_start in enumerate(start_points):
        if time.time() > deadline:
            break
            
        try:
            res = minimize(
                param_obj, 
                p_start, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                # Reconstruct f
                f_vals = basis @ res.x
                f_vals = np.maximum(0, f_vals)
                best_f = f_vals.copy()
                
        except Exception:
            pass
            
    # If we have time, do a local refinement on the full 600-dimensional vector?
    # The parametric solution is already a step function.
    # We can try a few steps of block coordinate descent on the best parametric solution.
    if best_f is not None and time.time() < deadline - 10:
        f = best_f.copy()
        current_val = best_val
        B = 20 # Larger block for final polish
        
        # Run a few iterations of block descent
        steps = 0
        while time.time() < deadline - 5 and steps < 50:
            steps += 1
            start_idx = np.random.randint(0, N - B)
            idxs = np.arange(start_idx, start_idx + B)
            x0 = f[idxs].copy()
            
            def local_obj(x):
                f_new = f.copy()
                f_new[idxs] = x
                return _autoconv_peak(f_new)
                
            try:
                res = minimize(
                    local_obj, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(0, None)] * B,
                    options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-6}
                )
                if res.fun < current_val:
                    f[idxs] = res.x
                    current_val = res.fun
                    if current_val < best_val:
                        best_val = current_val
                        best_f = f.copy()
            except Exception:
                pass
                
    if best_f is None:
        # Fallback to uniform
        best_f = np.ones(N)
        
    return list(best_f)

if __name__ == "__main__":
    result = construct()
    print(f"Result: {len(result)} values, sum={sum(result):.4f}")
