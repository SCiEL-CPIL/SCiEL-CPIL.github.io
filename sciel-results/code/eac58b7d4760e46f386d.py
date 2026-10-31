import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
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

def smooth_max_logsumexp(x, tau=1.0):
    """Smooth approximation of max(x) using log-sum-exp."""
    m = np.max(x)
    if m < 1e-30:
        return 1e-30
    return m + tau * np.log(np.sum(np.exp((x - m) / tau)))

def smooth_ratio(h, tau=0.01):
    """Smooth version of the ratio for gradient-based optimization."""
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf_smooth = smooth_max_logsumexp(conv, tau=tau)
    if conv_inf_smooth < 1e-15:
        return 0.0
    return h2_sq / conv_inf_smooth

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

def find_sidon_sets(max_size=12, n_trials=20000, deadline=None):
    """Find large Sidon sets in [0, N-1]."""
    best_sets = []
    best_size = 0
    
    # Greedy Sidon set construction from each starting point
    for start in range(N):
        indices = [start]
        sums = {2 * start}
        for candidate in range(N):
            if candidate == start:
                continue
            # Check if adding candidate maintains Sidon property
            valid = True
            for existing in indices:
                s = existing + candidate
                if s in sums:
                    valid = False
                    break
            if valid:
                indices.append(candidate)
                sums.add(2 * candidate)
                for existing in indices[:-1]:
                    sums.add(existing + candidate)
        
        if len(indices) > best_size:
            best_size = len(indices)
            best_sets = [indices.copy()]
        elif len(indices) == best_size:
            best_sets.append(indices.copy())
    
    # Random search for larger Sidon sets
    for _ in range(n_trials):
        if deadline and time.time() > deadline:
            break
        size = min(best_size + 1, 15)
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
    
    # Smooth objective for gradient-based methods
    def objective_neg_smooth(h_flat, tau=0.01):
        h = np.maximum(h_flat, 0.0)
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        h = h / s
        h2 = np.sum(h**2)
        conv = fftconvolve(h, h, mode='full')
        cinf_smooth = smooth_max_logsumexp(conv, tau=tau)
        if cinf_smooth < 1e-15:
            return 1e10
        return -h2 / cinf_smooth
    
    # Try L-BFGS-B with smooth objective
    if time.time() < deadline:
        try:
            res = minimize(objective_neg_smooth, h, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
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
    
    # Nelder-Mead as backup
    if time.time() < deadline:
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
            
            res = minimize(objective_neg, best_h, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
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
    
    # Try different weight distributions
    for alpha in [0.1, 0.25, 0.5, 0.75, 0.9]:
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
    
    return best_h, best_score

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin for 150s limit
    
    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    def update_best(h, sc):
        nonlocal best_h, best_score
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
    
    # Phase 1: Find large Sidon sets
    sidon_sets, max_sidon_size = find_sidon_sets(max_size=12, n_trials=30000, deadline=deadline - 80.0)
    
    # Phase 2: Optimize weights on each Sidon set
    for support in sidon_sets[:15]:
        if time.time() > deadline - 30.0:
            break
        h_opt, sc = optimize_weights_for_support(support, deadline)
        if h_opt is not None and sc > best_score:
            update_best(h_opt, sc)
    
    # Phase 3: Try sparse patterns
    if time.time() < deadline - 15.0:
        for i in range(N):
            for j in range(i + 1, N):
                h = np.zeros(N)
                h[i] = 0.5
                h[j] = 0.5
                sc = compute_ratio(h)
                update_best(h, sc)
            if time.time() > deadline - 15.0:
                break
    
    # Phase 4: Final high-precision polish with smooth objective
    if time.time() < deadline - 5.0:
        try:
            def objective_neg_smooth(h_flat, tau=0.005):
                h = np.maximum(h_flat, 0.0)
                s = np.sum(h)
                if s < 1e-15:
                    return 1e10
                h = h / s
                h2 = np.sum(h**2)
                conv = fftconvolve(h, h, mode='full')
                cinf_smooth = smooth_max_logsumexp(conv, tau=tau)
                if cinf_smooth < 1e-15:
                    return 1e10
                return -h2 / cinf_smooth
            
            res = minimize(objective_neg_smooth, best_h, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-14})
            h_opt = np.maximum(res.x, 0.0)
            s = np.sum(h_opt)
            if s > 1e-15:
                h_opt = h_opt / s
                sc = compute_ratio(h_opt)
                update_best(h_opt, sc)
        except Exception:
            pass
        
        # Final Nelder-Mead polish on exact objective
        if time.time() < deadline - 2.0:
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
                
                res = minimize(objective_neg, best_h, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
                h_opt = np.maximum(res.x, 0.0)
                s = np.sum(h_opt)
                if s > 1e-15:
                    h_opt = h_opt / s
                    sc = compute_ratio(h_opt)
                    update_best(h_opt, sc)
            except Exception:
                pass
    
    return list(best_h)
