"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as so

N = 50

# EVOLVE-BLOCK-START
def _compute_value(h):
    """Compute the objective value for given step heights."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return -1.0
    h = np.clip(h, 0.0, None)
    
    # Fast convolution
    g = np.convolve(h, h)
    m = len(g)
    
    # y = [0, g[0], ..., g[m-1], 0]
    # l2 = sum_{i=0}^m (w/3) * (y[i]^2 + y[i]y[i+1] + y[i+1]^2)
    # Note: y has length m+2. Indices 0..m+1.
    # y[0]=0, y[1..m]=g, y[m+1]=0.
    # The sum is over i in 0..m.
    # i=0: y[0]=0, y[1]=g[0]. Term: g[0]^2
    # i=m: y[m]=g[m-1], y[m+1]=0. Term: g[m-1]^2
    # i in 1..m-1: y[i]=g[i-1], y[i+1]=g[i]. Term: g[i-1]^2 + g[i-1]g[i] + g[i]^2
    
    # We can compute this efficiently:
    # Sum = g[0]^2 + g[m-1]^2 + sum_{j=0}^{m-2} (g[j]^2 + g[j]g[j+1] + g[j+1]^2)
    # Wait, the indices in the problem description:
    # y = [0, g[0], ..., g[m-1], 0]
    # i goes 0 to m.
    # i=0: y[0]=0, y[1]=g[0]. Term: 0 + 0 + g[0]^2 = g[0]^2
    # i=1: y[1]=g[0], y[2]=g[1]. Term: g[0]^2 + g[0]g[1] + g[1]^2
    # ...
    # i=m-1: y[m-1]=g[m-2], y[m]=g[m-1]. Term: g[m-2]^2 + g[m-2]g[m-1] + g[m-1]^2
    # i=m: y[m]=g[m-1], y[m+1]=0. Term: g[m-1]^2 + 0 + 0 = g[m-1]^2
    
    # So the total sum is:
    # g[0]^2 + g[m-1]^2 + sum_{j=0}^{m-2} (g[j]^2 + g[j]g[j+1] + g[j+1]^2)
    # Let's check if this matches the direct calculation.
    # Direct:
    # sum_{i=0}^m (y_i^2 + y_i y_{i+1} + y_{i+1}^2)
    # = y_0^2 + y_m y_{m+1} ... no, just sum the terms.
    
    # Let's just use the vectorized calculation as it is fast enough for N=50.
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    
    if norm_1 * norm_inf == 0:
        return -1.0
    return l2 / (norm_1 * norm_inf)

def _neg_value(h):
    """Negative of the value for minimization."""
    val = _compute_value(h)
    if val < 0:
        return 1e10
    return -val

def _hump_init():
    """Create a hump-shaped initial guess."""
    x = np.linspace(-1, 1, N)
    h = np.exp(-2 * x**2) # Gaussian hump
    h = h / np.max(h)
    return h

def _two_peak_init():
    """Create a two-peak initial guess."""
    x = np.linspace(-1, 1, N)
    h = np.exp(-4 * (x - 0.5)**2) + np.exp(-4 * (x + 0.5)**2)
    h = h / np.max(h)
    return h

def _ramp_init():
    """Create a ramp initial guess."""
    h = np.linspace(0, 1, N)
    return h

def _optimize_from_start(h0, deadline):
    """Run L-BFGS-B optimization from a given start."""
    if time.time() > deadline:
        return h0, _compute_value(h0)
    
    # L-BFGS-B is good for bound-constrained problems
    res = so.minimize(
        _neg_value, 
        h0, 
        method='L-BFGS-B', 
        bounds=[(0, None)] * N,
        options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8, 'maxls': 50}
    )
    
    h_opt = res.x
    val_opt = _compute_value(h_opt)
    
    return h_opt, val_opt

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    initializations = [
        _hump_init(),
        _two_peak_init(),
        _ramp_init(),
        np.ones(N), # Uniform (seed)
        np.array([0 if i < N//2 else 1 for i in range(N)]) # Step
    ]
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    for h0 in initializations:
        if time.time() > deadline:
            break
        
        h_opt, val_opt = _optimize_from_start(h0, deadline)
        
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt
    
    # If we have time left, do a final polish on the best solution found
    if time.time() < deadline - 5.0:
        # Try a small perturbation or re-run with tighter tolerance
        h_polish, val_polish = _optimize_from_start(best_h, deadline)
        if val_polish > best_val:
            best_val = val_polish
            best_h = h_polish
            
    return list(best_h)
# EVOLVE-BLOCK-END
