import numpy as np
from scipy.optimize import minimize
import time
import math

N = 50
DH = 1.0 / 20.0
OFFSET = 10

# Wall-clock deadline with safety margin
_DEADLINE = 115.0
_start = time.time()

def _elapsed():
    return time.time() - _start

def _time_left():
    return _DEADLINE - _start

def _autoconv(h):
    """Discrete autoconvolution sampled on the same grid (length N)."""
    h = np.asarray(h, dtype=float)
    full = np.convolve(h, h, mode='full')
    out = full[OFFSET:OFFSET + N]
    return DH * out

def _ratio(h):
    """Return the objective value R = ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 0.0
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 0.0:
        return 0.0
    return l2sq / (l1 * acmax)

def _smooth_max(x, sigma=0.01):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max == 0:
        return 0.0
    x_shifted = x - x_max
    # Avoid underflow/overflow in exp
    log_sum_exp = x_max + np.log(np.sum(np.exp(x_shifted / sigma))) * sigma
    return log_sum_exp

def _objective_complex(g_flat):
    """
    Objective to minimize: negative log of the ratio.
    g_flat is a real vector of length 2*N representing complex g.
    h = |g|^2.
    We normalize h to have L1 norm 1.
    """
    g = g_flat.reshape(N, 2)
    h = g[:, 0]**2 + g[:, 1]**2
    
    # Normalize h to have L1 norm 1
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 1e12
    h = h / l1
    
    l2sq = np.sum(h**2)
    
    # Compute autoconvolution
    ac = _autoconv(h)
    
    # Use smooth max for differentiability
    peak = _smooth_max(ac, sigma=0.01)
    
    if peak < 1e-12:
        return 1e12
    
    # Ratio = l2sq / (1 * peak) = l2sq / peak
    # We want to maximize Ratio, so minimize -log(Ratio)
    ratio = l2sq / peak
    if ratio <= 0:
        return 1e12
        
    return -math.log(ratio)

def _objective_complex_grad(g_flat):
    """
    Gradient of the objective function.
    This is numerically approximated or derived. 
    Given the complexity of the chain rule through smooth_max and normalization,
    we will use a numerical gradient or a simplified analytic one if possible.
    For robustness in this constrained time, we might rely on L-BFGS-B's 
    internal gradient estimation if we don't provide jac, but providing it is faster.
    However, deriving the full analytic gradient for smooth_max + normalization + |g|^2 
    is prone to errors. 
    Let's use a finite difference approach for the gradient to ensure correctness, 
    or simply let L-BFGS-B estimate it (slower but safer for this one-off).
    Actually, L-BFGS-B with 'jac=None' uses finite differences. With N=50 (100 vars), 
    it's 100 function calls per gradient step. This might be too slow for 115s.
    
    Let's try to provide an analytic gradient for the *main* terms and ignore the 
    smooth_max derivative complexity by using a larger sigma or just finite diff.
    
    Alternative: Use COBYLA or Nelder-Mead? No, L-BFGS is better.
    
    Let's implement the gradient analytically for the non-smooth-max part first? 
    No, the smooth max is key.
    
    Let's stick to finite differences for the gradient but limit iterations.
    Or, better: Use the analytic gradient of the *exact* max (subgradient) and rely 
    on the smoothness of the other terms? No, max is not smooth.
    
    Let's use a simple finite difference for the gradient in the objective function 
    call, but optimize the number of evaluations.
    """
    # We will not provide an analytic jac to avoid derivation errors.
    # L-BFGS-B will estimate it. To speed this up, we keep the function evals cheap.
    return None

def _optimize_complex(g0, time_limit):
    """Run L-BFGS-B optimization on complex parameters."""
    start_time = time.time()
    
    # No bounds, as g can be any real number
    # Use finite difference gradient estimation
    result = minimize(
        _objective_complex,
        g0,
        method='L-BFGS-B',
        jac=None, # Use finite difference
        options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 1500, 'eps': 1e-5}
    )
    
    elapsed = time.time() - start_time
    if elapsed > time_limit:
        # Return h corresponding to initial g0 if timeout
        g = g0.reshape(N, 2)
        h = g[:, 0]**2 + g[:, 1]**2
        return h
    
    g_opt = result.x.reshape(N, 2)
    h_opt = g_opt[:, 0]**2 + g_opt[:, 1]**2
    return h_opt

def _sparsify_and_optimize(h, K, time_limit):
    """Keep top K elements, set others to 0, and re-optimize."""
    h_sparse = h.copy()
    # Find top K indices
    indices = np.argsort(h_sparse)[-K:]
    mask = np.zeros(N, dtype=bool)
    mask[indices] = True
    h_sparse[~mask] = 0.0
    
    if np.sum(h_sparse) == 0:
        return h, 0.0
        
    # Initialize g from h_sparse
    g_real = np.sqrt(h_sparse)
    g_imag = np.zeros(N)
    g_flat = np.concatenate([g_real, g_imag])
    
    # Optimize
    h_opt = _optimize_complex(g_flat, time_limit)
    
    # Ensure sparsity is maintained? 
    # The optimizer might fill in the zeros. 
    # We should project back to the sparse support after optimization.
    h_opt[~mask] = 0.0
    
    return h_opt, _ratio(h_opt)

def construct():
    global _start
    _start = time.time()

    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    rng = np.random.default_rng(42)
    
    initializations = []
    
    # 1. Sparse Spikes (Sidon-like)
    # Try different sparse patterns
    for K in [5, 10, 15, 20]:
        # Random sparse
        indices = rng.choice(N, K, replace=False)
        h_sp = np.zeros(N)
        h_sp[indices] = 1.0
        initializations.append(h_sp)
        
        # Structured sparse (e.g., every 4th)
        if K <= N//4:
            h_struct = np.zeros(N)
            h_struct[::4] = 1.0
            initializations.append(h_struct)

    # 2. Uniform
    initializations.append(np.ones(N))
    
    # 3. Gaussian Lobes
    x = np.linspace(-0.25, 0.25, N)
    for w in [0.02, 0.05, 0.1]:
        h_g = np.exp(-x**2 / (2*w**2))
        initializations.append(h_g)
        
    # Bimodal
    h_bimodal = np.exp(-((x-0.1)**2)/0.01) + np.exp(-((x+0.1)**2)/0.01)
    initializations.append(h_bimodal)

    # 4. Random Smooth
    for seed in [1, 2, 3]:
        rng_local = np.random.RandomState(seed)
        h_raw = rng_local.random(N)
        kernel = np.array([1, 2, 3, 2, 1]) / 9.0
        h_smooth = np.convolve(h_raw, kernel, mode='same')
        initializations.append(h_smooth)

    # Sort initializations by initial score to prioritize good ones
    scored_inits = []
    for h0 in initializations:
        h0 = np.maximum(h0, 0.0)
        if np.sum(h0) == 0: continue
        h0 = h0 / np.sum(h0) # Normalize for consistent scale in init
        score = _ratio(h0)
        scored_inits.append((score, h0))
        
    scored_inits.sort(key=lambda x: -x[0])
    
    # Time budget allocation
    # We have ~115s. 
    # Let's try top 10 initializations with 5s each = 50s.
    # Then sparsify the best few with 10s each.
    
    time_per_opt = 4.0
    
    for score, h0 in scored_inits[:15]:
        if _time_left() < time_per_opt + 2.0:
            break
            
        # Convert h0 to g0
        g0_real = np.sqrt(h0)
        g0_imag = np.zeros(N)
        g0_flat = np.concatenate([g0_real, g0_imag])
        
        h_opt = _optimize_complex(g0_flat, time_per_opt)
        
        # Ensure non-negative and finite
        h_opt = np.maximum(h_opt, 0.0)
        
        score_opt = _ratio(h_opt)
        if score_opt > best_score:
            best_score = score_opt
            best_h = h_opt.copy()

    # Phase 2: Sparsification Polish
    # Try to improve the best_h by sparsifying it
    if _time_left() > 20.0:
        for K in [10, 15, 20, 25]:
            if _time_left() < 8.0:
                break
            h_sp, score_sp = _sparsify_and_optimize(best_h, K, 6.0)
            if score_sp > best_score:
                best_score = score_sp
                best_h = h_sp.copy()

    # Final safety check
    if best_h is None or not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        
    return [float(x) for x in best_h]

if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio(h))
    print("time =", time.time() - t0)
