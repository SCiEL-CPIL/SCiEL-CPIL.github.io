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

def is_sidon_set(indices):
    """Check if a set of indices forms a Sidon set (all pairwise sums are distinct)."""
    idx_set = set(indices)
    if len(idx_set) != len(indices):
        return False
    sums = set()
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def greedy_sidon(max_size=10, n=N, seed=0):
    """Greedy construction of a Sidon set."""
    rng = np.random.RandomState(seed)
    candidates = list(range(n))
    rng.shuffle(candidates)
    sidon = []
    for c in candidates:
        test = sidon + [c]
        if is_sidon_set(test):
            sidon.append(c)
            if len(sidon) >= max_size:
                break
    return sidon

def find_sidon_sets(n=N, max_size=10, num_trials=200):
    """Find multiple Sidon sets of various sizes."""
    sidon_sets = []
    for size in [3, 4, 5, 6, 7, 8, 9, 10]:
        for trial in range(num_trials // 8):
            s = greedy_sidon(max_size=size, n=n, seed=trial * 100 + size)
            if len(s) == size:
                sidon_sets.append(s)
    return sidon_sets

def optimize_weights_for_support(support, time_budget=30.0):
    """Optimize weights on a given support to maximize the ratio."""
    k = len(support)
    if k == 0:
        return None, 0.0
    
    def objective(w):
        h = np.zeros(N)
        h[support] = w
        return -compute_ratio(h)
    
    # Start with uniform weights
    w0 = np.ones(k) / k
    
    # Try L-BFGS-B
    try:
        result = minimize(objective, w0, method='L-BFGS-B',
                          bounds=[(0, 1)] * k,
                          options={'maxiter': 500, 'ftol': 1e-12})
        if result.success and result.fun < -0.5:
            h = np.zeros(N)
            h[support] = np.maximum(result.x, 0)
            sc = compute_ratio(h)
            return h, sc
    except Exception:
        pass
    
    # Try Nelder-Mead
    try:
        result = minimize(objective, w0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        if result.fun < -0.5:
            h = np.zeros(N)
            h[support] = np.maximum(result.x, 0)
            sc = compute_ratio(h)
            return h, sc
    except Exception:
        pass
    
    return None, 0.0

def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)

    def update_best(h, sc):
        nonlocal best_h, best_score
        if sc > best_score:
            best_score = sc
            best_h = h.copy()

    # Phase 1: Quick baselines
    h_delta = np.zeros(N); h_delta[0] = 1.0
    update_best(h_delta, compute_ratio(h_delta))

    h_unif = np.ones(N)
    update_best(h_unif, compute_ratio(h_unif))

    # Phase 2: Find Sidon sets and optimize weights on each
    sidon_sets = find_sidon_sets(N, max_size=10, num_trials=100)
    
    # Sort by size (larger Sidon sets likely better)
    sidon_sets.sort(key=lambda s: -len(s))
    
    for sidon in sidon_sets:
        if time.time() > deadline - 30.0:
            break
        h_opt, sc = optimize_weights_for_support(sidon, time_budget=20.0)
        if h_opt is not None:
            update_best(h_opt, sc)
    
    # Phase 3: Local refinement from best
    if time.time() < deadline - 15.0:
        h_start = best_h.copy()
        try:
            def obj_full(h_flat):
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
            
            res = minimize(obj_full, h_start, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            h_opt = np.maximum(res.x, 0.0)
            s = np.sum(h_opt)
            if s > 1e-15:
                h_opt = h_opt / s
                sc_opt = compute_ratio(h_opt)
                update_best(h_opt, sc_opt)
        except Exception:
            pass

    # Phase 4: Try some random sparse supports as backup
    if time.time() < deadline - 10.0:
        rng = np.random.RandomState(123)
        for trial in range(50):
            if time.time() > deadline - 10.0:
                break
            k = rng.randint(3, 11)
            support = list(rng.choice(N, size=k, replace=False))
            h_opt, sc = optimize_weights_for_support(support, time_budget=5.0)
            if h_opt is not None:
                update_best(h_opt, sc)

    return list(best_h)
