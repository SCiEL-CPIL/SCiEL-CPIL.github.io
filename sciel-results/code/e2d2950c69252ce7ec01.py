import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # Safety margin before 120s limit

def _score(h):
    """Compute the score: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    # Scale invariance: we can normalize to L1=1 to avoid overflow/underflow issues in conv
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    max_val = np.max(conv)
    if max_val <= 0:
        return 0.0
    return l2_sq / max_val

def _score_neg(h):
    return -_score(h)

def _generate_sidon_set(n, target_size, seed=42):
    """
    Greedy generation of a Sidon set (B2 set) of size 'target_size' within [0, n).
    A Sidon set has distinct pairwise sums (or differences).
    """
    rng = np.random.RandomState(seed)
    best_set = None
    best_size = 0
    
    # Try multiple random restarts to find a large Sidon set
    for attempt in range(50):
        if time.time() > DEADLINE - 10:
            break
        s = [0]
        # To make it faster, we can shuffle the candidate order slightly or just use greedy smallest-first
        # Greedy smallest-first usually yields good Sidon sets
        sums = set()
        # Pre-compute potential sums? No, just check on the fly.
        # For n=50, k~10, this is very fast.
        
        current_size = 1
        while current_size < target_size:
            found = False
            # Try candidates in increasing order to keep the set compact? 
            # Or random? Compact sets often have better autocorrelation properties?
            # Actually, for Sidon, "spread out" is better to avoid collisions.
            # Let's try a mix: random permutation of candidates.
            candidates = list(range(1, n))
            rng.shuffle(candidates)
            
            for cand in candidates:
                if cand in s:
                    continue
                # Check if adding cand creates duplicate sums
                # Sums of pairs (i, j) with i <= j
                # New sums: cand + s_i for all i in s, and cand + cand
                ok = True
                new_sums = set(sums)
                for idx in s:
                    ns = cand + idx
                    if ns in new_sums:
                        ok = False
                        break
                    new_sums.add(ns)
                if ok:
                    ns = cand + cand
                    if ns in new_sums:
                        ok = False
                    else:
                        new_sums.add(ns)
                
                if ok:
                    s.append(cand)
                    sums = new_sums
                    current_size += 1
                    found = True
                    break
            
            if not found:
                break
        
        if current_size > best_size:
            best_size = current_size
            best_set = s[:]
            if best_size >= target_size:
                break
        
    if best_set is None:
        best_set = list(range(min(target_size, n)))
    return best_set[:target_size]

def construct():
    start_time = time.time()
    
    # 1. Generate a pool of candidate supports based on Sidon sets of various sizes
    # Theoretical max size of Sidon set in [1..N] is approx sqrt(N). For N=50, sqrt(50)~7.
    # But we can have "almost" Sidon or just sparse sets.
    # Let's try sizes from 5 to 15.
    
    best_h = np.ones(N) / N  # Uniform fallback
    best_score = _score(best_h)
    
    candidate_supports = []
    
    # Generate Sidon sets
    for k in range(5, 16):
        if time.time() - start_time > DEADLINE - 20:
            break
        sidon = _generate_sidon_set(N, k, seed=k*10)
        if len(sidon) >= 5:
            candidate_supports.append(sidon)
            
    # Also add some random sparse sets
    rng = np.random.RandomState(123)
    for i in range(10):
        if time.time() - start_time > DEADLINE - 20:
            break
        k = rng.randint(5, 15)
        idx = rng.choice(N, k, replace=False)
        candidate_supports.append(idx.tolist())
        
    # If we didn't get good Sidon sets, add a default sparse set
    if not candidate_supports:
        candidate_supports.append(list(range(0, N, 5)))
        
    # 2. Optimize amplitudes for each support
    # We will use L-BFGS-B on the sub-vector of amplitudes.
    # To speed this up, we can define a wrapper that maps the sub-vector to the full vector.
    
    for support in candidate_supports:
        if time.time() - start_time > DEADLINE - 5:
            break
            
        m = len(support)
        if m < 2:
            continue
            
        # Initial guess: equal amplitudes
        h_full = np.zeros(N)
        h_full[support] = 1.0 / m
        
        def objective_sub(x):
            """Objective function for amplitudes on the support."""
            x = np.maximum(x, 0.0)
            h_temp = np.zeros(N)
            h_temp[support] = x
            return -_score(h_temp)
        
        try:
            res = minimize(objective_sub, 
                           np.ones(m) / m, 
                           method='L-BFGS-B', 
                           bounds=[(0, None)] * m,
                           options={'maxiter': 100, 'gtol': 1e-8, 'maxls': 20})
            
            h_opt_sub = np.maximum(res.x, 0.0)
            h_opt_full = np.zeros(N)
            h_opt_full[support] = h_opt_sub
            
            s_opt = _score(h_opt_full)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt_full.copy()
                
        except Exception:
            pass

    # 3. Global Polish
    # Take the best sparse solution and allow all 50 elements to move slightly.
    # This can break the strict support constraint and potentially find a better local maximum.
    if time.time() - start_time < DEADLINE - 10:
        try:
            res_global = minimize(_score_neg, 
                                  best_h, 
                                  method='L-BFGS-B', 
                                  bounds=[(0, None)] * N,
                                  options={'maxiter': 200, 'gtol': 1e-10, 'maxls': 20})
            h_global = np.maximum(res_global.x, 0.0)
            s_global = _score(h_global)
            if s_global > best_score:
                best_score = s_global
                best_h = h_global.copy()
        except Exception:
            pass

    # 4. Final Coordinate Descent Polish (Very fast, handles non-smooth max better sometimes)
    if time.time() - start_time < DEADLINE - 2:
        h_curr = best_h.copy()
        curr_score = _score(h_curr)
        
        # Try small perturbations
        for step in [0.01, 0.005, 0.001]:
            if time.time() - start_time > DEADLINE - 2:
                break
            improved = True
            while improved and time.time() - start_time < DEADLINE - 2:
                improved = False
                for i in range(N):
                    # Try increasing
                    h_trial = h_curr.copy()
                    h_trial[i] += step
                    h_trial = np.maximum(h_trial, 0.0)
                    s = _score(h_trial)
                    if s > curr_score + 1e-12:
                        h_curr = h_trial
                        curr_score = s
                        improved = True
                        continue
                    
                    # Try decreasing
                    h_trial = h_curr.copy()
                    h_trial[i] -= step
                    h_trial = np.maximum(h_trial, 0.0)
                    s = _score(h_trial)
                    if s > curr_score + 1e-12:
                        h_curr = h_trial
                        curr_score = s
                        improved = True
                        
        if curr_score > best_score:
            best_score = curr_score
            best_h = h_curr.copy()

    return best_h.tolist()
