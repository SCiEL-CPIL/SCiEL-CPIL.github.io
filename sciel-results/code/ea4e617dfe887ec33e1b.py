import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 50
DELTA = 0.5 / N

def _score_raw(h):
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

def _find_sidon_set(n_elements, n_positions, rng):
    """Find a Sidon-like set: all pairwise sums (with replacement) are distinct."""
    for _ in range(2000):
        s = set()
        valid = True
        chosen = []
        for _ in range(n_elements):
            found = False
            for _ in range(50):
                x = rng.integers(0, n_positions)
                if x in chosen:
                    continue
                # Check if x creates duplicate sums
                new_sums = set()
                conflict = False
                for y in chosen:
                    s_val = x + y
                    if s_val in s:
                        conflict = True
                        break
                    new_sums.add(s_val)
                if not conflict:
                    s_val_2 = 2 * x
                    if s_val_2 in s:
                        conflict = True
                    else:
                        new_sums.add(s_val_2)
                
                if conflict:
                    continue
                
                s.update(new_sums)
                chosen.append(x)
                found = True
                break
            if not found:
                valid = False
                break
        if valid and len(chosen) == n_elements:
            return np.array(chosen)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    # --- Phase 1: Generate diverse sparse initializations with Sidon sets ---
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Try different sparsity levels with Sidon-like supports
    best_sidon_supports = []
    for k in range(3, 18):
        found_any = False
        for attempt in range(5):
            sidon = _find_sidon_set(k, N, rng)
            if sidon is not None:
                h = np.zeros(N)
                h[sidon] = 1.0
                candidates.append(h)
                
                # Geometric weighting
                weights = np.exp(np.linspace(0, 1, k))
                h2 = np.zeros(N)
                h2[sidon] = weights
                candidates.append(h2)
                
                # Random weights
                for _ in range(3):
                    h3 = np.zeros(N)
                    h3[sidon] = rng.random(k) + 0.1
                    candidates.append(h3)
                
                best_sidon_supports.append(tuple(sidon))
                found_any = True
                break
        if not found_any:
            # Fallback to random sparse
            for _ in range(3):
                h = np.zeros(N)
                idx = rng.choice(N, k, replace=False)
                h[idx] = rng.random(k) + 0.05
                candidates.append(h)
    
    # Random sparse (not necessarily Sidon)
    for k in range(3, 25):
        for _ in range(20):
            h = np.zeros(N)
            idx = rng.choice(N, k, replace=False)
            h[idx] = rng.random(k) + 0.05
            candidates.append(h)
    
    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline - 40:
            break
        if np.sum(h) > 0:
            s = _score_raw(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()
    
    # --- Phase 2: Optimize weights for sparse supports ---
    def neg_obj_weights(weights, support):
        h = np.zeros(N)
        h[support] = weights
        l1 = np.sum(weights)
        if l1 == 0:
            return 0
        l2_sq = np.sum(weights**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv == 0:
            return 0
        return -(l2_sq / max_conv)
    
    # Collect unique supports from candidates and explicit Sidon sets
    supports = set()
    for h in candidates:
        mask = h > 1e-10
        if mask.sum() > 0 and mask.sum() < N:
            supports.add(tuple(np.where(mask)[0]))
    
    # Add explicit Sidon supports
    for k in range(3, 15):
        sidon = _find_sidon_set(k, N, rng)
        if sidon is not None:
            supports.add(tuple(sidon))
    
    # Optimize weights for each support
    optimized_supports = []
    for support in list(supports)[:60]:  # Limit number of optimizations
        if time.time() > deadline - 20:
            break
        
        k = len(support)
        if k > 20:
            continue
            
        bounds = [(0.01, 10.0)] * k
        
        # Multiple random starts
        for _ in range(5):
            if time.time() > deadline - 15:
                break
            w0 = rng.random(k) + 0.1
            try:
                res = minimize(
                    lambda w: neg_obj_weights(w, support),
                    w0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-12}
                )
                if res.success:
                    h_candidate = np.zeros(N)
                    h_candidate[support] = res.x
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate.copy()
                        optimized_supports.append(support)
            except Exception:
                pass
    
    # --- Phase 3: Full optimization with SLSQP ---
    def neg_obj_full(h):
        l1 = np.sum(h)
        if l1 == 0:
            return 0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv == 0:
            return 0
        return -(l2_sq / max_conv)
    
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
        {'type': 'ineq', 'fun': lambda h: h}
    ]
    bounds = [(0.0, 1.0)] * N
    
    h_start = best_h / np.sum(best_h)
    
    try:
        res = minimize(neg_obj_full, h_start, method='SLSQP', bounds=bounds,
                       constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
        if res.success:
            s_candidate = _score_raw(res.x)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = res.x.copy()
    except Exception:
        pass
    
    # --- Phase 4: Differential evolution for global search ---
    if time.time() < deadline - 25:
        def de_obj(h):
            l1 = np.sum(h)
            if l1 == 0:
                return 0
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            max_conv = np.max(h_conv)
            if max_conv == 0:
                return 0
            return -(l2_sq / max_conv)
        
        try:
            res = differential_evolution(
                de_obj,
                bounds=[(0, 1)] * N,
                maxiter=30,
                popsize=20,
                tol=1e-8,
                seed=42,
                polish=False
            )
            if res.success:
                s_candidate = _score_raw(res.x)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 5:
        h_start = best_h / np.sum(best_h)
        try:
            res = minimize(neg_obj_full, h_start, method='SLSQP', bounds=bounds,
                           constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-14})
            if res.success:
                s_candidate = _score_raw(res.x)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = res.x.copy()
        except Exception:
            pass
    
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) > 1e-10:
        best_h = best_h / np.sum(best_h)
    
    return list(best_h)
