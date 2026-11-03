import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 50
DELTA = 1.0 / N

def _autoconv_sup(h):
    """Compute sup norm of autoconvolution of step function with heights h."""
    n = len(h)
    num_points = 2 * n - 1
    ff_vals = np.zeros(num_points)
    
    for k in range(num_points):
        total = 0.0
        i_min = max(0, k - (n - 1))
        i_max = min(n - 1, k)
        for i in range(i_min, i_max + 1):
            j_min = max(0, k - i - 1)
            j_max = min(n - 1, k - i + 1)
            if j_min > j_max:
                continue
            js = np.arange(j_min, j_max + 1)
            lo = np.maximum(i, k - js)
            hi = np.minimum(i + 1, k - js + 1)
            overlaps = np.maximum(0, hi - lo)
            total += np.sum(h[i] * h[js] * overlaps)
        ff_vals[k] = total * DELTA
    
    return np.max(ff_vals)

def _autoconv_all(h):
    """Compute all autoconvolution values at breakpoints."""
    n = len(h)
    num_points = 2 * n - 1
    ff_vals = np.zeros(num_points)
    
    for k in range(num_points):
        total = 0.0
        i_min = max(0, k - (n - 1))
        i_max = min(n - 1, k)
        for i in range(i_min, i_max + 1):
            j_min = max(0, k - i - 1)
            j_max = min(n - 1, k - i + 1)
            if j_min > j_max:
                continue
            js = np.arange(j_min, j_max + 1)
            lo = np.maximum(i, k - js)
            hi = np.minimum(i + 1, k - js + 1)
            overlaps = np.maximum(0, hi - lo)
            total += np.sum(h[i] * h[js] * overlaps)
        ff_vals[k] = total * DELTA
    
    return ff_vals

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = _autoconv_sup(h)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def _smooth_autoconv_sup(h, beta=50.0):
    """Compute smooth approximation of sup norm using log-sum-exp."""
    ff_vals = _autoconv_all(h)
    max_val = np.max(ff_vals)
    if max_val == 0:
        return 0.0
    return max_val + np.log(np.sum(np.exp((ff_vals - max_val) * beta))) / beta

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

def generate_sidon_set(n, size):
    """Generate a Sidon (Mian-Chowla) set."""
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
        if len(sidon) >= size:
            break
    return sidon

def _iterative_reweighting(h0, max_iter=200, tol=1e-8, time_limit=None, start_time=None):
    """Iterative reweighting on fixed support to optimize heights."""
    h = np.maximum(h0, 0).copy()
    support = np.where(h > 1e-10)[0]
    
    if len(support) == 0:
        return h, _ratio(h)
    
    best_h = h.copy()
    best_score = _ratio(h)
    
    for it in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Find the current peak
        ff_vals = _autoconv_all(h)
        peak_idx = np.argmax(ff_vals)
        
        sum_h2 = np.sum(h**2)
        sum_h = np.sum(h)
        ff_peak = ff_vals[peak_idx]
        
        if sum_h < 1e-15 or ff_peak < 1e-15:
            break
        
        k = peak_idx
        n = len(h)
        dff_dh = np.zeros(n)
        for i in range(n):
            if h[i] < 1e-15:
                continue
            j_min = max(0, k - i - 1)
            j_max = min(n - 1, k - i + 1)
            if j_min > j_max:
                continue
            js = np.arange(j_min, j_max + 1)
            lo = np.maximum(i, k - js)
            hi = np.minimum(i + 1, k - js + 1)
            overlaps = np.maximum(0, hi - lo)
            dff_dh[i] = 2.0 * DELTA * np.sum(h[js] * overlaps)
        
        grad_log = 2.0 * h / sum_h2 - 1.0 / sum_h - dff_dh / ff_peak
        
        eta = 0.1
        for _ in range(10):
            h_new = h * np.exp(eta * grad_log)
            h_new = np.maximum(h_new, 0)
            
            s_new = np.sum(h_new)
            if s_new < 1e-15:
                break
            h_new = h_new / s_new
            
            score_new = _ratio(h_new)
            if score_new > best_score + 1e-12:
                h = h_new
                best_h = h.copy()
                best_score = score_new
                eta = min(eta * 1.5, 1.0)
                break
            else:
                eta /= 2.0
        
        if it > 0 and abs(best_score - _ratio(h)) < tol:
            break
    
    return best_h, best_score

def _coordinate_descent(h0, max_iter=500, tol=1e-10, time_limit=None, start_time=None):
    """Coordinate descent polish: optimize one coordinate at a time."""
    h = np.maximum(h0, 0).copy()
    best_score = _ratio(h)
    n = len(h)
    
    for it in range(max_iter):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        improved = False
        for i in range(n):
            if h[i] < 1e-15:
                continue
            
            for delta in [0.01, 0.001, 0.0001, -0.01, -0.001, -0.0001]:
                h_new = h.copy()
                h_new[i] = max(0, h[i] + delta)
                
                s = np.sum(h_new)
                if s < 1e-15:
                    continue
                h_new = h_new / s
                
                score = _ratio(h_new)
                if score > best_score + 1e-12:
                    h = h_new
                    best_score = score
                    improved = True
                    break
        
        if not improved:
            break
    
    return h, best_score

def _objective_sparse(weights, support):
    """Objective for sparse optimization: minimize M/Q (equivalent to maximizing Q/M with S=1)."""
    w = np.array(weights, dtype=float)
    if np.any(w < 0):
        return 1e10
    s = np.sum(w)
    if s < 1e-15:
        return 1e10
    w = w / s
    
    q = np.sum(w**2)
    if q < 1e-15:
        return 1e10
    
    # Build full vector for autoconv
    h = np.zeros(N)
    for i, idx in enumerate(support):
        h[idx] = w[i]
    
    sup_ff = _autoconv_sup(h)
    if sup_ff < 1e-15:
        return 1e10
    
    return sup_ff / q

def _optimize_sparse_support(support, time_limit=None, start_time=None):
    """Optimize heights on a given sparse support using differential evolution."""
    K = len(support)
    if K < 2:
        return None, 0.0
    
    bounds = [(0.0, 1.0)] * K
    
    try:
        res = differential_evolution(
            _objective_sparse, bounds, args=(support,),
            maxiter=80, popsize=15, tol=1e-6, seed=42, polish=False,
            update_interval=1
        )
        
        w_opt = res.x
        s = np.sum(w_opt)
        if s < 1e-15:
            return None, 0.0
        w_opt = w_opt / s
        
        h = np.zeros(N)
        for i, idx in enumerate(support):
            h[idx] = w_opt[i]
        
        score = _ratio(h)
        return h, score
    except Exception:
        return None, 0.0

def _smooth_objective(x, support):
    """Smooth objective for L-BFGS-B on sparse support."""
    w = np.maximum(x, 0)
    s = np.sum(w)
    if s < 1e-15:
        return 0.0
    w = w / s
    
    h = np.zeros(N)
    for i, idx in enumerate(support):
        h[idx] = w[i]
    
    return -_smooth_ratio(h, beta=30.0)

def _refine_sparse(h0, support, time_limit=None, start_time=None):
    """Refine a sparse solution using L-BFGS-B on the support."""
    K = len(support)
    w0 = np.array([h0[idx] for idx in support], dtype=float)
    s0 = np.sum(w0)
    if s0 < 1e-15:
        return h0, _ratio(h0)
    w0 = w0 / s0
    
    try:
        result = minimize(
            _smooth_objective, w0, args=(support,),
            method='L-BFGS-B',
            bounds=[(0, None)] * K,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        w_opt = np.maximum(result.x, 0)
        s = np.sum(w_opt)
        if s < 1e-15:
            return h0, _ratio(h0)
        w_opt = w_opt / s
        
        h = np.zeros(N)
        for i, idx in enumerate(support):
            h[idx] = w_opt[i]
        
        score = _ratio(h)
        return h, score
    except Exception:
        return h0, _ratio(h0)

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Known 2-element solution
    h2 = np.zeros(N)
    h2[0] = 2.0 / 3.0
    h2[1] = 1.0 / 3.0
    score2 = _ratio(h2)
    if score2 > best_score:
        best_score = score2
        best_h = h2.copy()
    
    np.random.seed(42)
    
    # Phase 1: Global search over sparse supports using differential evolution
    # Try various Sidon set sizes and random sparse supports
    candidate_supports = []
    
    # Sidon sets
    for size in [3, 4, 5, 6, 7, 8]:
        s = generate_sidon_set(N, size)
        if len(s) >= size:
            candidate_supports.append(s)
    
    # Random sparse supports
    for _ in range(20):
        num_ones = np.random.randint(3, 12)
        indices = np.random.choice(N, num_ones, replace=False)
        candidate_supports.append(sorted(indices.tolist()))
    
    # Structured patterns
    candidate_supports.append(list(range(0, N, 2)))  # every other
    candidate_supports.append(list(range(0, N, 3)))  # every third
    candidate_supports.append(list(range(0, N, 4)))  # every fourth
    
    # Process each support with differential evolution (time-limited)
    for support in candidate_supports:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        K = len(support)
        if K < 2 or K > 12:
            continue
        
        h_cand, score_cand = _optimize_sparse_support(support, time_limit=DEADLINE - 10, start_time=start_time)
        
        if h_cand is not None and score_cand > best_score:
            best_score = score_cand
            best_h = h_cand.copy()
    
    # Phase 2: Refine the best solution found
    if time.time() - start_time < DEADLINE - 15:
        # Extract support from best_h
        support_best = sorted(np.where(best_h > 1e-10)[0].tolist())
        
        if len(support_best) >= 2:
            h_refined, score_refined = _refine_sparse(best_h, support_best, time_limit=DEADLINE - 5, start_time=start_time)
            if score_refined > best_score:
                best_score = score_refined
                best_h = h_refined.copy()
    
    # Phase 3: Iterative reweighting polish
    if time.time() - start_time < DEADLINE - 5:
        try:
            h_rew, score_rew = _iterative_reweighting(
                best_h, max_iter=150, tol=1e-8,
                time_limit=DEADLINE - 3, start_time=start_time
            )
            if score_rew > best_score:
                best_score = score_rew
                best_h = h_rew.copy()
        except Exception:
            pass
    
    # Phase 4: Coordinate descent final polish
    if time.time() - start_time < DEADLINE - 2:
        try:
            h_polished, score_polished = _coordinate_descent(
                best_h, max_iter=200, tol=1e-10,
                time_limit=DEADLINE - 1, start_time=start_time
            )
            if score_polished > best_score:
                best_score = score_polished
                best_h = h_polished.copy()
        except Exception:
            pass
    
    return best_h.tolist()
