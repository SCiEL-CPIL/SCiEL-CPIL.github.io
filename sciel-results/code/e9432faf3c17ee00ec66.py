import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    """Compute the raw score: L2_sq / (L1 * autoconv_max)"""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _soft_max(x, alpha=50.0):
    """Smooth approximation of max using log-sum-exp."""
    m = np.max(x)
    return m + np.log(np.sum(np.exp(alpha * (x - m)))) / alpha

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # =====================================================================
    # PHASE 1: Parametric initializations
    # =====================================================================
    # Uniform
    h_init = np.ones(N) / N
    s = _score_raw(h_init)
    if s > best_score:
        best_score = s
        best_h = h_init.copy()
        
    # Gaussians
    for c in np.linspace(3, N-3, 10):
        for w in np.linspace(1, N/3, 10):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_gauss /= np.sum(h_gauss)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # Triangular
    for c in np.linspace(3, N-3, 10):
        for w in np.linspace(1, N, 10):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                h_tri /= np.sum(h_tri)
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()

    # =====================================================================
    # PHASE 2: Support pattern search with amplitude optimization
    # =====================================================================
    
    def optimize_amplitudes(support_indices, n_restarts=20):
        """Given a set of support indices, optimize the amplitudes."""
        k = len(support_indices)
        if k < 2:
            return None, 0.0
        
        # Use log-parameterization: a_i = exp(x_i) / sum(exp(x_j))
        # This ensures positivity and sums to 1 automatically
        
        def neg_obj(x):
            a = np.exp(x)
            a = a / np.sum(a)
            h = np.zeros(N)
            h[support_indices] = a
            s = _score_raw(h)
            if s <= 0:
                return 0.0
            return -s
        
        # Multiple random restarts
        best_local = None
        best_local_score = -1.0
        
        for _ in range(n_restarts):
            x0 = np.random.randn(k) * 0.5
            try:
                res = minimize(neg_obj, x0, method='L-BFGS-B', 
                              options={'maxiter': 500, 'ftol': 1e-12})
                if res.fun < 0:
                    a = np.exp(res.x)
                    a = a / np.sum(a)
                    h = np.zeros(N)
                    h[support_indices] = a
                    s = _score_raw(h)
                    if s > best_local_score:
                        best_local_score = s
                        best_local = h.copy()
            except Exception:
                pass
        
        if best_local is not None:
            return best_local, best_local_score
        
        # Fallback: uniform amplitudes
        h = np.zeros(N)
        h[support_indices] = 1.0 / k
        s = _score_raw(h)
        return h, s
    
    all_indices = np.arange(N)
    
    # Try various support sizes
    support_sizes = [2, 3, 4, 5, 6, 7, 8, 10]
    
    for size in support_sizes:
        if time.time() > deadline - 30:
            break
        
        # Generate candidate support sets
        candidates = set()
        
        # Random subsets
        n_random = 300 if size <= 5 else 150
        for _ in range(n_random):
            subset = tuple(np.sort(np.random.choice(all_indices, size, replace=False)))
            candidates.add(subset)
        
        # Structured: evenly spaced
        even = tuple(np.linspace(0, N-1, size, dtype=int))
        candidates.add(even)
        
        # Structured: clustered at various positions
        for start in range(0, N - size + 1, max(1, (N - size) // 10)):
            candidates.add(tuple(range(start, start + size)))
        
        # Structured: pairs of clusters
        if size >= 4:
            half = size // 2
            for pos1 in range(0, N - half, max(1, N // 20)):
                for pos2 in range(pos1 + half + 1, N - half, max(1, N // 20)):
                    if pos2 - pos1 >= size:
                        c1 = tuple(range(pos1, pos1 + half))
                        c2 = tuple(range(pos2, pos2 + half))
                        if size - half > 0:
                            c2 = c2 + tuple(range(pos2 + half, pos2 + half + (size - half)))
                        candidates.add(c1 + c2)
        
        # Deduplicate and optimize
        for support in candidates:
            if time.time() > deadline - 20:
                break
            
            h_candidate, s_candidate = optimize_amplitudes(list(support))
            if h_candidate is not None and s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
    
    # =====================================================================
    # PHASE 3: Local refinement - perturb the support
    # =====================================================================
    
    if time.time() < deadline - 40:
        active = np.where(best_h > 1e-6)[0]
        
        # Try adding one more element
        for add_idx in all_indices:
            if add_idx in active:
                continue
            if time.time() > deadline - 20:
                break
            new_support = sorted(list(active) + [add_idx])
            h_candidate, s_candidate = optimize_amplitudes(new_support)
            if h_candidate is not None and s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
                active = np.where(best_h > 1e-6)[0]
        
        # Try removing one element from active
        if len(active) > 2:
            for rem_idx in active.copy():
                if time.time() > deadline - 20:
                    break
                new_support = [i for i in active if i != rem_idx]
                h_candidate, s_candidate = optimize_amplitudes(new_support)
                if h_candidate is not None and s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
                    active = np.where(best_h > 1e-6)[0]
    
    # =====================================================================
    # PHASE 4: Final continuous optimization on the full space
    # =====================================================================
    
    if time.time() < deadline - 15:
        h_start = best_h.copy()
        h_start[h_start < 1e-8] = 1e-8
        x0 = np.log(h_start)
        
        def neg_obj_full(x):
            h = np.exp(x)
            h = h / np.sum(h)
            s = _score_raw(h)
            if s <= 0:
                return 0.0
            return -s
        
        # Multiple restarts from different perturbations
        best_full = None
        best_full_score = -1.0
        
        restarts = [x0]
        for _ in range(10):
            restarts.append(x0 + np.random.randn(N) * 0.1)
        
        for x_start in restarts:
            if time.time() > deadline - 10:
                break
            try:
                res = minimize(neg_obj_full, x_start, method='L-BFGS-B', 
                              options={'maxiter': 1000, 'ftol': 1e-14})
                if res.fun < 0:
                    h_candidate = np.exp(res.x)
                    h_candidate = h_candidate / np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_full_score:
                        best_full_score = s_candidate
                        best_full = h_candidate.copy()
            except Exception:
                pass
        
        if best_full is not None and best_full_score > best_score:
            best_score = best_full_score
            best_h = best_full
    
    return list(best_h)
