import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def smooth_ratio(h, beta=50.0):
    """Smooth surrogate using log-sum-exp for the max(conv) term."""
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0, None
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    # log-sum-exp for smooth max
    c = conv.max()
    if c < 1e-15:
        return 0.0, None
    log_sum_exp = c + np.log(np.sum(np.exp(conv - c)))
    smooth_max = log_sum_exp / beta  # This isn't quite right, let me use proper soft-max
    # Actually, soft-max: (1/beta) * log(sum(exp(beta * conv)))
    beta_conv = beta * conv
    beta_max = beta_conv.max()
    soft_max = (beta_max + np.log(np.sum(np.exp(beta_conv - beta_max)))) / beta
    if soft_max < 1e-15:
        return 0.0, None
    return h2_sq / soft_max, h_norm

def is_sidon_set(indices):
    """Check if a set of indices forms a Sidon set (all pairwise sums distinct)."""
    sums = set()
    n = len(indices)
    for i in range(n):
        for j in range(i, n):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def find_sidon_sets(n_trials=30000, deadline=None):
    """Find large Sidon sets in [0, N-1]."""
    best_sets = []
    best_size = 0
    
    # Greedy Sidon set construction from each starting point
    for start in range(N):
        indices = [start]
        for candidate in range(N):
            if candidate == start:
                continue
            trial = indices + [candidate]
            if is_sidon_set(trial):
                indices = trial
        if len(indices) > best_size:
            best_size = len(indices)
            best_sets = [indices.copy()]
        elif len(indices) == best_size:
            best_sets.append(indices.copy())
    
    # Random search for larger Sidon sets
    for _ in range(n_trials):
        if deadline and time.time() > deadline:
            break
        size = min(best_size + 1, 14)
        if size <= 2:
            continue
        trial = np.random.choice(N, size=size, replace=False).tolist()
        if is_sidon_set(trial):
            if size > best_size:
                best_size = size
                best_sets = [trial.copy()]
            else:
                best_sets.append(trial.copy())
    
    return best_sets, best_size

def optimize_weights_for_support(support, deadline):
    """Optimize weights on a given support to maximize the ratio."""
    k = len(support)
    if k == 0:
        return None, 0.0
    
    # Start with uniform weights
    h = np.zeros(N)
    h[support] = 1.0 / k
    best_h = h.copy()
    best_score = compute_ratio(h)
    
    def objective_neg(h_flat):
        h = np.maximum(h_flat, 0.0)
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        h = h / s
        h2 = np.sum(h**2)
        conv = fftconvolve(h, h, mode='full')
        cinf = np.max(conv)
        if cinf < 1e-15:
            return 1e10
        return -h2 / cinf
    
    # Local optimization from uniform start using L-BFGS-B
    if time.time() < deadline:
        try:
            res = minimize(objective_neg, h, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            h_opt = np.maximum(res.x, 0.0)
            s = np.sum(h_opt)
            if s > 1e-15:
                h_opt = h_opt / s
                sc = compute_ratio(h_opt)
                if sc > best_score:
                    best_score = sc
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Also try Nelder-Mead
    if time.time() < deadline:
        try:
            res = minimize(objective_neg, best_h, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            h_opt = np.maximum(res.x, 0.0)
            s = np.sum(h_opt)
            if s > 1e-15:
                h_opt = h_opt / s
                sc = compute_ratio(h_opt)
                if sc > best_score:
                    best_score = sc
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Try different weight distributions (power-law, geometric, etc.)
    for alpha in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        if time.time() > deadline:
            break
        h_trial = np.zeros(N)
        positions = np.arange(k)
        weights = (positions + 1) ** alpha
        weights = weights / np.sum(weights)
        h_trial[support] = weights
        sc = compute_ratio(h_trial)
        if sc > best_score:
            best_score = sc
            best_h = h_trial.copy()
    
    # Geometric weights
    for r in [0.8, 0.9, 0.95, 1.1, 1.25]:
        if time.time() > deadline:
            break
        h_trial = np.zeros(N)
        positions = np.arange(k)
        weights = r ** positions
        weights = weights / np.sum(weights)
        h_trial[support] = weights
        sc = compute_ratio(h_trial)
        if sc > best_score:
            best_score = sc
            best_h = h_trial.copy()
    
    # Bell-shaped weights
    for width in [0.5, 1.0, 1.5, 2.0]:
        if time.time() > deadline:
            break
        h_trial = np.zeros(N)
        positions = np.arange(k)
        center = k / 2.0
        weights = np.exp(-((positions - center) ** 2) / (2 * width ** 2))
        weights = weights / np.sum(weights)
        h_trial[support] = weights
        sc = compute_ratio(h_trial)
        if sc > best_score:
            best_score = sc
            best_h = h_trial.copy()
    
    return best_h, best_score

def local_search_swap(h, deadline, max_iters=2000):
    """Local search: try swapping support elements or adjusting weights."""
    h = h.copy()
    best_score = compute_ratio(h)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        
        # Try adjusting individual weights
        nonzero_idx = np.where(h > 1e-10)[0]
        if len(nonzero_idx) == 0:
            break
        
        for idx in nonzero_idx:
            # Try increasing
            h_trial = h.copy()
            h_trial[idx] *= 1.5
            sc = compute_ratio(h_trial)
            if sc > best_score:
                h = h_trial
                best_score = sc
                improved = True
                break
            # Try decreasing
            h_trial = h.copy()
            h_trial[idx] *= 0.5
            if np.sum(h_trial) < 1e-15:
                continue
            sc = compute_ratio(h_trial)
            if sc > best_score:
                h = h_trial
                best_score = sc
                improved = True
                break
        
        if not improved:
            # Try adding a new element
            for j in range(N):
                if h[j] > 1e-10:
                    continue
                h_trial = h.copy()
                h_trial[j] = np.max(h) * 0.1
                sc = compute_ratio(h_trial)
                if sc > best_score:
                    h = h_trial
                    best_score = sc
                    improved = True
                    break
                # Try removing
                if h[j] > 1e-10:
                    h_trial = h.copy()
                    h_trial[j] = 0.0
                    if np.sum(h_trial) < 1e-15:
                        continue
                    sc = compute_ratio(h_trial)
                    if sc > best_score:
                        h = h_trial
                        best_score = sc
                        improved = True
                        break
        
        if not improved:
            break
    
    return h, best_score

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # Safety margin (150s limit, leave 20s buffer)
    
    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    def update_best(h, sc):
        nonlocal best_h, best_score
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
    
    # Phase 1: Find large Sidon sets
    sidon_sets, max_sidon_size = find_sidon_sets(n_trials=20000, deadline=deadline - 80.0)
    
    # Phase 2: Optimize weights on each Sidon set
    # Sort by size descending, then try top ones
    sidon_sets_sorted = sorted(sidon_sets, key=len, reverse=True)
    for support in sidon_sets_sorted[:25]:
        if time.time() > deadline - 40.0:
            break
        h_opt, sc = optimize_weights_for_support(support, deadline)
        if h_opt is not None and sc > best_score:
            update_best(h_opt, sc)
    
    # Phase 3: Local search from best solution
    if time.time() < deadline - 20.0:
        h_ls, sc_ls = local_search_swap(best_h, deadline - 10.0, max_iters=1000)
        if sc_ls > best_score:
            update_best(h_ls, sc_ls)
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            def objective_neg(h_flat):
                h = np.maximum(h_flat, 0.0)
                s = np.sum(h)
                if s < 1e-15:
                    return 1e10
                h = h / s
                h2 = np.sum(h**2)
                conv = fftconvolve(h, h, mode='full')
                cinf = np.max(conv)
                if cinf < 1e-15:
                    return 1e10
                return -h2 / cinf
            
            # Use tight tolerances for final polish
            res = minimize(objective_neg, best_h, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14})
            h_opt = np.maximum(res.x, 0.0)
            s = np.sum(h_opt)
            if s > 1e-15:
                h_opt = h_opt / s
                sc = compute_ratio(h_opt)
                update_best(h_opt, sc)
        except Exception:
            pass
        
        # Another Nelder-Mead polish
        if time.time() < deadline - 2.0:
            try:
                def objective_neg2(h_flat):
                    h = np.maximum(h_flat, 0.0)
                    s = np.sum(h)
                    if s < 1e-15:
                        return 1e10
                    h = h / s
                    h2 = np.sum(h**2)
                    conv = fftconvolve(h, h, mode='full')
                    cinf = np.max(conv)
                    if cinf < 1e-15:
                        return 1e10
                    return -h2 / cinf
                
                res = minimize(objective_neg2, best_h, method='Nelder-Mead',
                              options={'maxiter': 8000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
                h_opt = np.maximum(res.x, 0.0)
                s = np.sum(h_opt)
                if s > 1e-15:
                    h_opt = h_opt / s
                    sc = compute_ratio(h_opt)
                    update_best(h_opt, sc)
            except Exception:
                pass
    
    return list(best_h)
