"""
Optimized approach for the third autocorrelation inequality.
Uses direct quadratic form optimization with active-set max,
smooth surrogate for initialization, and iterative local refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

N = 400
DEADLINE = 145.0  # Safety margin before 150s kill

# Precompute the convolution matrix for each lag
# For np.convolve(h, h, mode='full'), result[k] = sum_j h[j] * h[k-j]
# where 0 <= j < N and 0 <= k-j < N
# So result[k] = sum_{j=max(0,k-N+1)}^{min(N-1,k)} h[j] * h[k-j]
# This is a quadratic form: result[k] = h^T C_k h where C_k[i,j] = 1 if i+j == k

def _build_conv_matrices():
    """Build matrices C_k for each lag k in [0, 2N-2]."""
    C = {}
    for k in range(2 * N - 1):
        # C_k[i,j] = 1 if i + j == k, with 0 <= i,j < N
        # We only need the upper triangle or use a sparse-like approach
        # For efficiency, we'll compute h @ C_k @ h directly using the structure
        # C_k is a permutation-like matrix: C_k[i, k-i] = 1 for valid i
        # So h^T C_k h = sum_i h[i] * h[k-i] for valid i
        # This is just the cross-correlation at lag k
        C[k] = k
    return C

def _eval_value(h):
    """Compute the objective value (lower is better)."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(~np.isfinite(h)):
        return np.inf
    sh = float(np.sum(h))
    if abs(sh) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    max_conv = float(np.max(conv))
    val = abs(2.0 * N * max_conv / (sh * sh))
    return val

def _direct_objective_and_grad(h):
    """
    Compute the exact objective and its gradient.
    The objective is f(h) = |2N * max_k(h^T C_k h) / (sum(h))^2|
    
    We use the fact that at the optimum, the max is achieved at some lag k*,
    and we can optimize the quadratic form for that lag.
    """
    h = np.asarray(h, dtype=np.float64)
    sh = float(np.sum(h))
    if abs(sh) < 1e-12:
        return 1e10, np.zeros_like(h)
    
    conv = np.convolve(h, h)
    max_idx = np.argmax(conv)
    max_conv = conv[max_idx]
    
    # Objective: f = |2N * max_conv / sh^2|
    # Since we're taking abs, and 2N/sh^2 > 0, f = 2N * |max_conv| / sh^2
    # But max_conv could be negative, so we need to be careful.
    # Actually, the max over signed entries means we take the largest value,
    # which could be negative if all are negative. But typically we want
    # to minimize the absolute value, so let's handle the sign properly.
    
    # Let's define g = 2N * max_conv / sh^2 (without abs)
    # f = |g|
    # If g > 0, f = g, grad_f = grad_g
    # If g < 0, f = -g, grad_f = -grad_g
    
    # grad_g = 2N * (grad_max_conv / sh^2 - 2 * max_conv * sh / sh^4)
    #        = 2N * (grad_max_conv / sh^2 - 2 * max_conv / sh^3)
    # where grad_max_conv is the gradient of max_conv w.r.t. h
    
    # max_conv = h^T C_k h for k = max_idx
    # grad_max_conv = 2 * C_k @ h (since C_k is symmetric)
    
    # C_k @ h: (C_k @ h)[i] = sum_j C_k[i,j] * h[j] = h[k-i] if 0 <= k-i < N, else 0
    k = max_idx
    Ck_h = np.zeros(N)
    for i in range(N):
        j = k - i
        if 0 <= j < N:
            Ck_h[i] = h[j]
    
    # Since C_k is symmetric (C_k[i,j] = C_k[j,i] when i+j=k), 
    # grad(h^T C_k h) = 2 * C_k @ h
    grad_max_conv = 2.0 * Ck_h
    
    g = 2.0 * N * max_conv / (sh * sh)
    grad_g = 2.0 * N * (grad_max_conv / (sh * sh) - 2.0 * max_conv / (sh ** 3))
    
    f = abs(g)
    if g >= 0:
        grad_f = grad_g
    else:
        grad_f = -grad_g
    
    return f, grad_f

def _smooth_objective(h, beta=100.0):
    """Smooth surrogate using log-sum-exp for the max."""
    h = np.asarray(h, dtype=np.float64)
    s = float(np.sum(h))
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    max_c = np.max(conv)
    shifted = conv - max_c
    # Numerically stable log-sum-exp
    log_sum_exp = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
    val = abs(2.0 * N * log_sum_exp / (s * s))
    if not np.isfinite(val):
        return 1e10
    return val

def _smooth_grad(h, beta=100.0):
    """Gradient of the smooth surrogate."""
    h = np.asarray(h, dtype=np.float64)
    s = float(np.sum(h))
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    
    conv = np.convolve(h, h)
    max_c = np.max(conv)
    shifted = conv - max_c
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    
    # grad_lse[i] = 2 * sum_{j} weights[i+j] * h[j] for valid j
    # Vectorized: for each i, sum over j where i+j is a valid lag
    # weights has length 2N-1, indexed by lag k = i+j
    # For a given i, j ranges over [max(0, 0-i), min(N-1, 2N-2-i)]
    # But since i in [0, N-1] and j in [0, N-1], k = i+j in [i, i+N-1]
    # So grad_lse[i] = 2 * sum_{j=0}^{N-1} weights[i+j] * h[j]
    
    # Use matrix multiplication
    idx_i = np.arange(N)[:, None]
    idx_j = np.arange(N)[None, :]
    W = weights[idx_i + idx_j]  # W[i,j] = weights[i+j]
    grad_lse = 2.0 * (W @ h)
    
    lse = max_c + np.log(np.sum(weights)) / beta
    f = 2.0 * N * lse / (s * s)
    
    df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s ** 3)
    
    if f >= 0:
        grad = df_dh
    else:
        grad = -df_dh
    
    return grad

def _local_refine(h0, max_iter=200, tol=1e-10):
    """Local refinement using L-BFGS-B with direct objective."""
    h0 = np.asarray(h0, dtype=np.float64).copy()
    best_h = h0.copy()
    best_val = _eval_value(h0)
    
    def obj(x):
        v, _ = _direct_objective_and_grad(x)
        return v
    
    def grad(x):
        _, g = _direct_objective_and_grad(x)
        return g
    
    try:
        result = minimize(
            obj,
            h0,
            jac=grad,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': tol, 'maxfun': 500}
        )
        h_candidate = result.x
        v = _eval_value(h_candidate)
        if v < best_val and np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
            best_val = v
            best_h = h_candidate.copy()
    except Exception:
        pass
    
    return best_h, best_val

def _smooth_refine(h0, beta=100.0, max_iter=100):
    """Refine using smooth surrogate."""
    h0 = np.asarray(h0, dtype=np.float64).copy()
    best_h = h0.copy()
    best_val = _eval_value(h0)
    
    try:
        result = minimize(
            lambda x: _smooth_objective(x, beta),
            h0,
            jac=lambda x: _smooth_grad(x, beta),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-8, 'maxfun': 200}
        )
        h_candidate = result.x
        v = _eval_value(h_candidate)
        if v < best_val and np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
            best_val = v
            best_h = h_candidate.copy()
    except Exception:
        pass
    
    return best_h, best_val

def _diverse_init(rng):
    """Generate diverse initial points."""
    initial_points = []
    t = np.linspace(0, 1, N, endpoint=False)
    i_arr = np.arange(N)
    
    # Seed
    initial_points.append(np.ones(N).astype(float))
    
    # Random uniform
    for _ in range(10):
        initial_points.append(rng.uniform(-1, 1, N))
    
    # Bimodal: mostly zero with random spikes
    for _ in range(10):
        h = np.zeros(N)
        k = rng.randint(50, 350)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        initial_points.append(h)
    
    # Alternating signs
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    initial_points.append(h_alt)
    
    # Low-frequency cosines and sines
    for freq in [1, 2, 3, 4, 5, 7, 10, 15, 20]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        initial_points.append(np.sin(2 * np.pi * freq * t))
    
    # Parameterized bump profiles
    for center in [N//8, N//4, N//2, 3*N//4, 7*N//8]:
        for width in [20, 50, 100, 150]:
            h_bump = np.exp(-((i_arr - center)**2) / (2 * width**2))
            initial_points.append(h_bump)
            initial_points.append(-h_bump)
            h_double = np.exp(-((i_arr - center)**2) / (2 * width**2)) - \
                       np.exp(-((i_arr - (N-1-center))**2) / (2 * width**2))
            initial_points.append(h_double)
    
    # Random Fourier synthesis
    for n_comp in [3, 5, 8]:
        for _ in range(5):
            freqs = rng.randint(1, 40, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.5, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.cos(2 * np.pi * f * t + p)
            initial_points.append(h_fourier)
    
    # Structured patterns: blocks of +1 and -1
    for block_size in [20, 50, 100]:
        n_blocks = N // block_size
        for _ in range(3):
            h_blocks = np.zeros(N)
            for b in range(n_blocks):
                sign = rng.choice([-1, 1])
                h_blocks[b*block_size:(b+1)*block_size] = sign
            initial_points.append(h_blocks)
    
    # Smooth random via Gaussian filter
    for _ in range(10):
        h_smooth = rng.uniform(-1, 1, N)
        h_smooth = gaussian_filter1d(h_smooth, sigma=rng.uniform(5, 20))
        initial_points.append(h_smooth)
    
    # Logistic map based patterns
    for r in [3.5, 3.7, 3.9, 4.0]:
        x = rng.uniform(0, 1)
        h_log = np.zeros(N)
        for i in range(N):
            h_log[i] = x
            x = r * x * (1 - x)
        initial_points.append(h_log)
        initial_points.append(-h_log)
    
    return initial_points

def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_val = _eval_value(best_h)  # 2.0
    
    def elapsed():
        return time.time() - t0
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initial points
    initial_points = _diverse_init(rng)
    
    # Evaluate and sort initial points
    init_values = []
    for init_h in initial_points:
        val_init = _eval_value(init_h)
        if np.isfinite(val_init):
            init_values.append((val_init, init_h))
    init_values.sort(key=lambda x: x[0])
    
    # Take top initial points for optimization
    top_inits = [h for _, h in init_values[:30]]
    
    # Optimize from each initial point
    for init_h in top_inits:
        if elapsed() > DEADLINE - 15:
            break
        
        # Normalize to have non-zero sum
        s = np.sum(init_h)
        if abs(s) < 1e-10:
            init_h = init_h + 0.1
        
        # Try smooth refinement first (faster, good for initialization)
        for beta in [50.0, 100.0]:
            if elapsed() > DEADLINE - 10:
                break
            h_cand, v_cand = _smooth_refine(init_h, beta=beta, max_iter=50)
            if v_cand < best_val:
                best_val = v_cand
                best_h = h_cand.copy()
        
        # Then direct refinement
        if elapsed() > DEADLINE - 8:
            continue
        h_cand, v_cand = _local_refine(best_h if v_cand < best_val else init_h, max_iter=150)
        if v_cand < best_val:
            best_val = v_cand
            best_h = h_cand.copy()
    
    # Final polish from best solution
    if elapsed() < DEADLINE - 5:
        for beta_polish in [200.0, 500.0]:
            if elapsed() > DEADLINE - 3:
                break
            h_cand, v_cand = _smooth_refine(best_h, beta=beta_polish, max_iter=30)
            if v_cand < best_val:
                best_val = v_cand
                best_h = h_cand.copy()
        
        if elapsed() < DEADLINE - 3:
            h_cand, v_cand = _local_refine(best_h, max_iter=100, tol=1e-12)
            if v_cand < best_val:
                best_val = v_cand
                best_h = h_cand.copy()
    
    # Final safety: ensure finite and non-zero sum
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h
