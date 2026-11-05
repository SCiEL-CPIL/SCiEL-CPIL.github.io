"""Hybrid approach: direct quadratic form optimization with active-set max, smooth surrogate initialization, and local refinement."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from scipy.special import eval_chebyt

N = 400
DEADLINE = 145.0  # Safety margin before 150s kill

# Precompute Chebyshev basis
_x = np.linspace(-1, 1, N)
_basis_cache = {}

def _get_basis(degree):
    if degree in _basis_cache:
        return _basis_cache[degree]
    B = np.zeros((degree + 1, N))
    for k in range(degree + 1):
        B[k] = eval_chebyt(k, _x)
    _basis_cache[degree] = B
    return B

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

def _build_conv_matrix(lag):
    """Build the N x N symmetric matrix C such that h^T C h = conv(h,h)[lag]."""
    C = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        j = lag - i
        if 0 <= j < N:
            C[i, j] = 1.0
    return C

def _direct_objective_and_grad(h, active_lag=None):
    """Compute objective and gradient using direct quadratic form with active max lag."""
    h = np.asarray(h, dtype=np.float64)
    s = float(np.sum(h))
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(h)
    
    conv = np.convolve(h, h)
    if active_lag is None:
        active_lag = int(np.argmax(conv))
    
    C = _build_conv_matrix(active_lag)
    qh = float(h @ C @ h)
    f = 2.0 * N * qh / (s * s)
    
    # Gradient of h^T C h is (C + C.T)h = 2Ch since C is symmetric
    grad_qh = 2.0 * (C @ h)
    grad = 2.0 * N * grad_qh / (s * s) - 4.0 * N * qh / (s * s * s) * np.ones(N)
    
    if f < 0:
        f = -f
        grad = -grad
    
    if not np.isfinite(f):
        return 1e10, np.zeros_like(h)
    
    return f, grad

def _smooth_objective(x, beta=80.0):
    """Smooth surrogate using log-sum-exp for the max."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    max_c = np.max(conv)
    shifted = conv - max_c
    log_sum_exp = max_c + np.log(np.sum(np.exp(beta * shifted))) / beta
    val = abs(2.0 * N * log_sum_exp / (s * s))
    if not np.isfinite(val):
        return 1e10
    return val

def _smooth_grad(x, beta=80.0):
    """Gradient of the smooth surrogate."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    
    conv = np.convolve(h, h)
    max_c = np.max(conv)
    shifted = conv - max_c
    weights = np.exp(beta * shifted)
    wsum = np.sum(weights)
    if wsum < 1e-300:
        return np.zeros_like(h)
    weights /= wsum
    
    # Vectorized gradient computation
    idx_i = np.arange(N)[:, None]
    idx_j = np.arange(N)[None, :]
    W = weights[idx_i + idx_j]
    grad_lse = 2.0 * (W @ h)
    
    lse = max_c + np.log(wsum) / beta
    f = 2.0 * N * lse / (s * s)
    
    df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s) * np.ones(N)
    
    if f >= 0:
        grad = df_dh
    else:
        grad = -df_dh
    
    return grad

def _direct_refine(h0, max_iters=200, tol=1e-10):
    """Direct refinement using active-set method on the max autocorrelation lag."""
    h = h0.copy().astype(np.float64)
    s = float(np.sum(h))
    if abs(s) < 1e-10:
        return h
    
    for it in range(max_iters):
        conv = np.convolve(h, h)
        k_star = int(np.argmax(conv))
        
        C = _build_conv_matrix(k_star)
        qh = float(h @ C @ h)
        s = float(np.sum(h))
        if abs(s) < 1e-10:
            break
        
        f = 2.0 * N * qh / (s * s)
        grad = 2.0 * N * (2.0 * C @ h) / (s * s) - 4.0 * N * qh / (s * s * s) * np.ones(N)
        
        if f < 0:
            grad = -grad
        
        gnorm = np.linalg.norm(grad)
        if gnorm < tol:
            break
        
        # Backtracking line search
        alpha = 1.0
        f_current = abs(f)
        for ls_iter in range(30):
            h_new = h - alpha * grad
            s_new = float(np.sum(h_new))
            if abs(s_new) < 1e-10:
                alpha *= 0.5
                continue
            conv_new = np.convolve(h_new, h_new)
            k_new = int(np.argmax(conv_new))
            C_new = _build_conv_matrix(k_new)
            qh_new = float(h_new @ C_new @ h_new)
            f_new = abs(2.0 * N * qh_new / (s_new * s_new))
            
            if f_new < f_current - 1e-12 * alpha * gnorm * gnorm:
                h = h_new
                break
            alpha *= 0.5
        else:
            break
    
    return h

def _eigen_init():
    """Generate initial points using eigenvalue decomposition of the max convolution matrix.
    
    The key insight: we want to minimize max_k (h^T C_k h) / (sum(h))^2.
    If we fix sum(h) = 1, we minimize max_k h^T C_k h.
    A good starting point is to spread energy across eigenmodes that don't concentrate
    in any single C_k.
    """
    inits = []
    t = np.linspace(0, 1, N, endpoint=False)
    
    # Multi-frequency cosine combinations
    for n_freq in range(2, 8):
        for _ in range(3):
            freqs = np.random.randint(1, 30, n_freq)
            phases = np.random.uniform(0, 2 * np.pi, n_freq)
            amps = np.random.uniform(0.5, 2.0, n_freq)
            h = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h += a * np.cos(2 * np.pi * f * t + p)
            if abs(np.sum(h)) > 1e-6:
                inits.append(h)
    
    # Structured anti-correlated patterns
    for period in [7, 11, 13, 17, 23, 29, 31, 37]:
        h = np.zeros(N)
        for i in range(N):
            phase = 2 * np.pi * i / period
            h[i] = np.cos(phase) + 0.3 * np.cos(2 * phase)
        if abs(np.sum(h)) > 1e-6:
            inits.append(h)
    
    # Random with zero-mean constraint
    for _ in range(20):
        h = np.random.randn(N)
        h -= np.mean(h)
        h += 0.1  # Ensure non-zero sum
        inits.append(h)
    
    return inits

def _multi_start_local(init_points, budget_time=60.0):
    """Run local refinement from multiple starting points."""
    t0 = time.time()
    best_h = None
    best_val = np.inf
    
    for init_h in init_points:
        if time.time() - t0 > budget_time:
            break
        try:
            h_refined = _direct_refine(init_h.copy(), max_iters=150, tol=1e-11)
            v = _eval_value(h_refined)
            if v < best_val and np.all(np.isfinite(h_refined)) and abs(np.sum(h_refined)) > 1e-10:
                best_val = v
                best_h = h_refined.copy()
        except Exception:
            pass
    
    return best_h, best_val

def construct():
    t0 = time.time()
    best_h = np.ones(N)
    best_val = _eval_value(best_h)  # 2.0
    
    def elapsed():
        return time.time() - t0
    
    rng = np.random.RandomState(42)
    np.random.seed(42)
    
    # --- Generate diverse initial points ---
    initial_points = []
    
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
    
    # Low-frequency cosines
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 4, 5, 7, 10, 15, 20]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        initial_points.append(np.sin(2 * np.pi * freq * t))
    
    # Parameterized bump profiles
    i_arr = np.arange(N)
    for center in [N//8, N//4, N//2, 3*N//4, 7*N//8]:
        for width in [20, 50, 100, 150]:
            h_bump = np.exp(-((i_arr - center)**2) / (2 * width**2))
            initial_points.append(h_bump)
            initial_points.append(-h_bump)
            h_double = np.exp(-((i_arr - center)**2) / (2 * width**2)) - np.exp(-((i_arr - (N-1-center)**2) / (2 * width**2)))
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
        
    # Chebyshev-based initializations with random coefficients
    for degree in [5, 10, 15, 20]:
        B = _get_basis(degree)
        for _ in range(5):
            coeffs = rng.uniform(-1, 1, degree + 1)
            h_cheb = B.T @ coeffs
            initial_points.append(h_cheb)
    
    # Eigen-based initializations
    initial_points.extend(_eigen_init())

    # --- Evaluate and sort initial points ---
    init_values = []
    for init_h in initial_points:
        val_init = _eval_value(init_h)
        init_values.append((val_init, init_h))
    init_values.sort(key=lambda x: x[0])
    
    # Take top 30 initial points for optimization
    top_inits = [h for _, h in init_values[:30]]
    
    # --- Phase 1: Smooth surrogate optimization for good starting points ---
    for init_h in top_inits:
        if elapsed() > DEADLINE - 50:
            break
        
        s = np.sum(init_h)
        if abs(s) < 1e-10:
            init_h = init_h + 0.1
        
        try:
            result = minimize(
                lambda x, b=80.0: _smooth_objective(x, b),
                init_h,
                jac=lambda x, b=80.0: _smooth_grad(x, b),
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 200, 'gtol': 1e-6}
            )
            
            if elapsed() > DEADLINE - 45:
                break
            
            h_candidate = result.x
            v = _eval_value(h_candidate)
            if v < best_val and np.all(np.isfinite(h_candidate)) and abs(np.sum(h_candidate)) > 1e-10:
                best_val = v
                best_h = h_candidate.copy()
        except Exception:
            pass
    
    # --- Phase 2: Direct active-set refinement from best and top candidates ---
    if elapsed() < DEADLINE - 30:
        try:
            h_refined = _direct_refine(best_h.copy(), max_iters=300, tol=1e-12)
            v = _eval_value(h_refined)
            if v < best_val and np.all(np.isfinite(h_refined)) and abs(np.sum(h_refined)) > 1e-10:
                best_val = v
                best_h = h_refined.copy()
        except Exception:
            pass
    
    # Refine from top 10 initial points
    for init_h in top_inits[:10]:
        if elapsed() > DEADLINE - 20:
            break
        try:
            h_refined = _direct_refine(init_h.copy(), max_iters=200, tol=1e-11)
            v = _eval_value(h_refined)
            if v < best_val and np.all(np.isfinite(h_refined)) and abs(np.sum(h_refined)) > 1e-10:
                best_val = v
                best_h = h_refined.copy()
        except Exception:
            pass
    
    # --- Phase 3: Final high-precision polish ---
    if elapsed() < DEADLINE - 10:
        try:
            h0 = best_h.copy()
            h_polished = _direct_refine(h0, max_iters=500, tol=1e-14)
            v = _eval_value(h_polished)
            if v < best_val and np.all(np.isfinite(h_polished)) and abs(np.sum(h_polished)) > 1e-10:
                best_val = v
                best_h = h_polished.copy()
        except Exception:
            pass
    
    # --- Phase 4: Local perturbation search around best ---
    if elapsed() < DEADLINE - 5:
        try:
            for _ in range(30):
                if elapsed() > DEADLINE - 3:
                    break
                noise = rng.normal(0, 0.01, N)
                h_perturbed = best_h + noise
                s_p = np.sum(h_perturbed)
                if abs(s_p) < 1e-10:
                    continue
                h_refined = _direct_refine(h_perturbed, max_iters=100, tol=1e-12)
                v = _eval_value(h_refined)
                if v < best_val and np.all(np.isfinite(h_refined)) and abs(np.sum(h_refined)) > 1e-10:
                    best_val = v
                    best_h = h_refined.copy()
        except Exception:
            pass
    
    # Final safety
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h
