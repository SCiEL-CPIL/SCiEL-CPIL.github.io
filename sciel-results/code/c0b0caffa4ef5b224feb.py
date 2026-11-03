import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 50
DELTA = 1.0 / N

def _precompute_overlaps():
    """Precompute overlap matrices for all k values."""
    overlaps = []
    for k in range(2 * N - 1):
        mat = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                lo = max(i, k - j)
                hi = min(i + 1, k - j + 1)
                ov = max(0.0, hi - lo)
                if ov > 0:
                    mat[i, j] = ov
        overlaps.append(mat)
    return overlaps

# Precompute once at module level
_OVERLAPS = _precompute_overlaps()

def _autoconv_values(h):
    """Compute all autoconvolution values at breakpoints using precomputed overlaps."""
    n = len(h)
    num_points = 2 * n - 1
    ff_vals = np.zeros(num_points)
    h_outer = h[:, None] * h[None, :]
    for k in range(num_points):
        ff_vals[k] = DELTA * np.sum(h_outer * _OVERLAPS[k])
    return ff_vals

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    ff_vals = _autoconv_values(h)
    sup_ff = np.max(ff_vals)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def _smooth_autoconv_sup(h, beta=50.0):
    """Smooth approximation of sup norm using log-sum-exp."""
    ff_vals = _autoconv_values(h)
    max_val = np.max(ff_vals)
    if max_val < 1e-15:
        return 1e-15
    shifted = (ff_vals - max_val) * beta
    # Clip to avoid overflow
    shifted = np.clip(shifted, -500, 500)
    return max_val + np.log(np.sum(np.exp(shifted))) / beta

def _smooth_ratio(h, beta=50.0):
    """Compute smooth ratio."""
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = _smooth_autoconv_sup(h, beta)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def _neg_smooth_ratio(x, beta=50.0):
    """Negative smooth ratio for minimization."""
    x = np.maximum(x, 0)
    s = np.sum(x)
    if s < 1e-15:
        return 0.0
    return -_smooth_ratio(x / s, beta)

def _neg_ratio(x):
    """Negative raw ratio for minimization."""
    x = np.maximum(x, 0)
    s = np.sum(x)
    if s < 1e-15:
        return 0.0
    return -_ratio(x / s)

def _differential_evolution_neg_ratio(x, beta=50.0):
    """For DE: use smooth version."""
    return _neg_smooth_ratio(x, beta)

def _sidon_sets(n, max_size):
    """Generate Sidon (Mian-Chowla) sets."""
    results = []
    # Greedy Mian-Chowla
    sidon = [0]
    used_sums = {0}
    for candidate in range(1, n):
        valid = True
        for i in sidon:
            if i + candidate in used_sums:
                valid = False
                break
        if valid:
            sidon.append(candidate)
            used_sums.add(candidate + candidate)
            for i in sidon[:-1]:
                used_sums.add(i + candidate)
        if len(sidon) >= max_size:
            break
    results.append(sidon[:max_size])
    
    # Random Sidon sets
    for trial in range(20):
        np.random.seed(trial * 7 + 13)
        s = []
        used = set()
        for _ in range(max_size):
            for _ in range(1000):
                c = np.random.randint(0, n)
                if c in used:
                    continue
                ok = True
                for i in s:
                    if i + c in used:
                        ok = False
                        break
                if ok:
                    s.append(c)
                    used.add(c)
                    used.add(c + c)
                    for i in s[:-1]:
                        used.add(i + c)
                    break
        if len(s) == max_size:
            results.append(s)
    return results

def _subgradient_step(h, beta=30.0, lr=0.1, time_limit=None, start_time=None, max_iter=200):
    """Subgradient-like optimization using smooth surrogate."""
    h = np.maximum(h, 0).copy()
    s = np.sum(h)
    if s < 1e-15:
        return h, _ratio(h)
    h = h / s
    
    best_h = h.copy()
    best_score = _ratio(h)
    
    for it in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Compute subgradient of -smooth_ratio
        # Use numerical gradient for robustness
        eps = 1e-5
        grad = np.zeros(N)
        base_val = _neg_smooth_ratio(h, beta)
        
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] += eps
            sp = np.sum(h_plus)
            if sp > 1e-15:
                val_plus = _neg_smooth_ratio(h_plus / sp, beta)
            else:
                val_plus = 0.0
            grad[i] = (val_plus - base_val) / eps
        
        # Normalize gradient (scale invariance means we project out the all-ones direction)
        grad_mean = np.mean(grad)
        grad = grad - grad_mean
        
        # Gradient descent step
        h_new = h - lr * grad
        h_new = np.maximum(h_new, 0)
        s_new = np.sum(h_new)
        if s_new < 1e-15:
            break
        h_new = h_new / s_new
        
        score_new = _ratio(h_new)
        if score_new > best_score:
            best_score = score_new
            best_h = h_new.copy()
            lr = min(lr * 1.1, 0.5)
        else:
            lr *= 0.8
        
        # Check convergence
        if lr < 1e-6:
            break
    
    return best_h, best_score

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    np.random.seed(42)
    
    # Phase 1: Generate diverse initializations
    initializations = []
    
    # Sidon sets of various sizes
    for size in [4, 5, 6, 7, 8, 10, 12]:
        sidon_sets = _sidon_sets(N, size)
        for s in sidon_sets[:5]:  # Limit number per size
            h_mc = np.zeros(N)
            for idx in s:
                h_mc[idx] = 1.0
            initializations.append(h_mc)
    
    # Random sparse vectors
    for _ in range(20):
        h_sp = np.zeros(N)
        num_ones = np.random.randint(3, 15)
        indices = np.random.choice(N, num_ones, replace=False)
        h_sp[indices] = np.random.uniform(0.3, 2.0, num_ones)
        initializations.append(h_sp)
    
    # Structured patterns
    t = np.linspace(-1, 1, N)
    for width in [0.2, 0.4, 0.6, 1.0]:
        initializations.append(np.exp(-t**2/width))
    initializations.append(np.maximum(1 - t**2, 0.01))
    initializations.append(np.abs(np.sin(np.pi * t)))
    initializations.append(np.abs(np.cos(np.pi * t)))
    
    # Bimodal
    initializations.append(np.maximum(1.0 + 2.0*np.exp(-((t-0.6)**2)/0.1) + 2.0*np.exp(-((t+0.6)**2)/0.1), 0.05))
    initializations.append(np.maximum(1.0 + 3.0*np.exp(-((t-0.5)**2)/0.05) + 3.0*np.exp(-((t+0.5)**2)/0.05), 0.05))
    
    # Edge spikes
    h_edges = np.zeros(N)
    h_edges[0] = 1.0
    h_edges[N-1] = 1.0
    initializations.append(h_edges)
    
    # Centered spike
    h_center = np.zeros(N)
    h_center[N//2] = 1.0
    initializations.append(h_center)
    
    # Two-point solutions (known to be good for small N)
    for i, j in [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (N//2-1, N//2), (N//2, N//2+1)]:
        h_two = np.zeros(N)
        h_two[i] = 1.0
        h_two[j] = 1.0
        initializations.append(h_two)
    
    # Random dense
    for _ in range(10):
        initializations.append(np.random.uniform(0, 1, N))
    
    # Phase 2: Optimize each initialization
    for h0 in initializations:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        h0 = np.maximum(h0, 0)
        s0 = np.sum(h0)
        if s0 < 1e-15:
            continue
        h0 = h0 / s0
        
        # Method 1: L-BFGS-B on smooth surrogate
        if time.time() - start_time < DEADLINE - 10:
            try:
                result = minimize(_neg_smooth_ratio, h0.copy(), method='L-BFGS-B',
                                 bounds=[(0, None)] * N,
                                 options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'beta': 30.0})
                
                h_candidate = np.maximum(result.x, 0)
                s = np.sum(h_candidate)
                if s > 1e-15:
                    h_candidate = h_candidate / s
                    sc = _ratio(h_candidate)
                    if sc > best_score:
                        best_score = sc
                        best_h = h_candidate.copy()
            except Exception:
                pass
        
        # Method 2: Subgradient step
        if time.time() - start_time < DEADLINE - 10:
            try:
                h_sg, score_sg = _subgradient_step(
                    h0, beta=30.0, lr=0.1,
                    time_limit=DEADLINE - 12, start_time=start_time, max_iter=100
                )
                if score_sg > best_score:
                    best_score = score_sg
                    best_h = h_sg.copy()
            except Exception:
                pass
    
    # Phase 3: Refine best solution with multiple methods
    if time.time() - start_time < DEADLINE - 5:
        # L-BFGS-B refinement with different betas
        for beta in [20, 40, 80, 100]:
            if time.time() - start_time > DEADLINE - 5:
                break
            try:
                def neg_smooth(x, b=beta):
                    return _neg_smooth_ratio(x, b)
                
                result = minimize(neg_smooth, best_h.copy(), method='L-BFGS-B',
                                 bounds=[(0, None)] * N,
                                 options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
                
                h_candidate = np.maximum(result.x, 0)
                s = np.sum(h_candidate)
                if s > 1e-15:
                    h_candidate = h_candidate / s
                    sc = _ratio(h_candidate)
                    if sc > best_score:
                        best_score = sc
                        best_h = h_candidate.copy()
            except Exception:
                pass
        
        # Coordinate descent polish
        if time.time() - start_time < DEADLINE - 3:
            h_cd = best_h.copy()
            for it in range(200):
                if time.time() - start_time > DEADLINE - 3:
                    break
                improved = False
                for i in range(N):
                    if h_cd[i] < 1e-15:
                        continue
                    for delta in [0.005, 0.001, 0.0001, -0.005, -0.001, -0.0001]:
                        h_new = h_cd.copy()
                        h_new[i] = max(0, h_cd[i] + delta)
                        s = np.sum(h_new)
                        if s < 1e-15:
                            continue
                        h_new = h_new / s
                        sc = _ratio(h_new)
                        if sc > best_score + 1e-12:
                            h_cd = h_new
                            best_score = sc
                            best_h = h_cd.copy()
                            improved = True
                            break
                if not improved:
                    break
    
    return best_h.tolist()
