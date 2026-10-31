import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

def is_sidon(A):
    """Check if A is a Sidon set (all sums a+b with a<=b are distinct)."""
    sums = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def find_sidon_set_greedy(n):
    """Greedy construction of a Sidon set."""
    A = []
    used_sums = set()
    for x in range(n):
        valid = True
        new_sums = []
        for a in A:
            s = a + x
            if s in used_sums:
                valid = False
                break
            new_sums.append(s)
        if 2 * x in used_sums:
            valid = False
        
        if valid:
            A.append(x)
            used_sums.update(new_sums)
            used_sums.add(2 * x)
    return A

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    l2_sq = np.sum(h * h)
    hh = np.convolve(h, h, mode='full')
    hh_max = np.max(hh)
    if hh_max < 1e-300:
        return 0.0
    return l2_sq / (l1 * hh_max)

def smooth_neg_ratio(params, beta=100.0):
    """Negative ratio with smooth max approximation for better gradient."""
    h = np.exp(params)
    h = h / np.sum(h)
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    l2_sq = np.sum(h * h)
    hh = np.convolve(h, h, mode='full')
    # Use log-sum-exp for smooth max
    hh_max_exact = np.max(hh)
    if hh_max_exact < 1e-300:
        return 0.0
    hh_shifted = hh - hh_max_exact
    # Ensure numerical stability
    hh_max_smooth = (1.0 / beta) * np.log(np.sum(np.exp(beta * hh_shifted))) + hh_max_exact
    if hh_max_smooth < 1e-300:
        return 0.0
    return -l2_sq / (l1 * hh_max_smooth)

def optimize_weights(h_init, time_budget, start_time):
    """Optimize weights using L-BFGS-B with smooth objective."""
    h_init = np.asarray(h_init, dtype=np.float64)
    h_pos = h_init + 1e-10
    x0 = np.log(h_pos)
    
    best_h = h_init.copy()
    best_val = compute_ratio(best_h)
    
    # Try a few different beta values
    betas = [50.0, 100.0, 200.0]
    
    for beta in betas:
        if time.time() - start_time > time_budget:
            break
            
        try:
            res = minimize(smooth_neg_ratio, x0, args=(beta,), method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12,
                                   'maxls': 20})
            h_opt = np.exp(res.x)
            h_opt = h_opt / np.sum(h_opt)
            val = compute_ratio(h_opt)
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
            
    return best_h, best_val

def local_search_support(h, compute_ratio_fn, time_limit, start_time):
    """Local search: try moving mass from one index to another to improve ratio.
    Focuses on sparse solutions."""
    h = h.copy()
    current_val = compute_ratio_fn(h)
    n_nonzero = np.sum(h > 1e-12)
    
    # Build list of indices with significant mass
    indices = np.where(h > 1e-12)[0]
    
    improved = True
    iters = 0
    while improved and time.time() - start_time < time_limit:
        improved = False
        iters += 1
        
        # Try moving mass from one index to another
        for i in range(len(indices)):
            if time.time() - start_time > time_limit:
                break
            a_idx = indices[i]
            for j in range(len(indices)):
                if i == j:
                    continue
                b_idx = indices[j]
                
                # Try moving a fraction of mass
                for frac in [0.1, 0.5, 1.0]:
                    delta = frac * h[a_idx]
                    if delta < 1e-15:
                        continue
                    new_h = h.copy()
                    new_h[a_idx] -= delta
                    new_h[b_idx] += delta
                    if new_h[a_idx] < -1e-15:
                        continue
                    new_h = np.maximum(new_h, 0.0)
                    val = compute_ratio_fn(new_h)
                    if val > current_val + 1e-12:
                        h = new_h
                        current_val = val
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        # Also try adding a new non-zero entry if we have room
        if not improved:
            for k in range(N):
                if h[k] > 1e-12:
                    continue
                # Try adding a small amount
                for amt in [0.01, 0.05]:
                    new_h = h.copy()
                    new_h[k] = amt
                    # Renormalize
                    new_h /= np.sum(new_h)
                    val = compute_ratio_fn(new_h)
                    if val > current_val + 1e-12:
                        h = new_h
                        current_val = val
                        improved = True
                        break
                if improved:
                    break
                    
    return h, current_val

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    
    def elapsed():
        return time.time() - start
    
    best_h = np.ones(N) / N
    best_val = compute_ratio(best_h)
    
    candidates = []
    
    # 1. Generate multiple Sidon set candidates
    rng = np.random.default_rng(42)
    
    # Greedy Sidon
    A_greedy = find_sidon_set_greedy(N)
    h_sidon = np.zeros(N)
    for idx in A_greedy:
        h_sidon[idx] = 1.0
    h_sidon /= np.sum(h_sidon)
    candidates.append(h_sidon)
    
    # Random Sidon sets
    best_A = A_greedy
    for _ in range(50): # Try 50 random restarts for Sidon sets
        if elapsed() > 5: # Time limit for init
            break
        size = rng.integers(8, 15)
        A_try = sorted(rng.choice(N, size, replace=False))
        if is_sidon(A_try):
            if len(A_try) > len(best_A):
                best_A = A_try
            h_rand_sidon = np.zeros(N)
            for idx in A_try:
                h_rand_sidon[idx] = 1.0
            h_rand_sidon /= np.sum(h_rand_sidon)
            candidates.append(h_rand_sidon)
            
    # Add best_A if different from greedy
    if len(best_A) != len(A_greedy) or set(best_A) != set(A_greedy):
        h_best_sidon = np.zeros(N)
        for idx in best_A:
            h_best_sidon[idx] = 1.0
        h_best_sidon /= np.sum(h_best_sidon)
        candidates.append(h_best_sidon)
        
    # 2. Other structured initializations
    # Uniform
    candidates.append(np.ones(N) / N)
    
    # Sparse random
    for k in [5, 10, 15]:
        for _ in range(5):
            h_rand = np.zeros(N)
            idx = rng.choice(N, size=k, replace=False)
            h_rand[idx] = 1.0 / k
            candidates.append(h_rand)
            
    # 3. Evaluate all candidates
    for h_cand in candidates:
        val = compute_ratio(h_cand)
        if val > best_val:
            best_val = val
            best_h = h_cand.copy()
            
    # 4. Refine top candidates
    scored = [(compute_ratio(h), h) for h in candidates]
    scored.sort(key=lambda x: -x[0])
    
    # Allocate remaining time
    time_elapsed = elapsed()
    time_remaining = DEADLINE - time_elapsed
    
    # Refine top 5 candidates
    n_refine = min(5, len(scored))
    time_per_refine = (time_remaining * 0.6) / n_refine
    
    for i in range(n_refine):
        if elapsed() > DEADLINE - 10:
            break
        h_init = scored[i][1]
        h_opt, val_opt = optimize_weights(h_init, time_per_refine, start)
        if val_opt > best_val:
            best_val = val_opt
            best_h = h_opt.copy()
            
    # 5. Local search on the best solution
    if elapsed() < DEADLINE - 5:
        h_ls, val_ls = local_search_support(best_h, compute_ratio, DEADLINE - 5 - elapsed(), start)
        if val_ls > best_val:
            best_val = val_ls
            best_h = h_ls.copy()
            
    # 6. Final polish
    if elapsed() < DEADLINE - 2:
        h_polish, val_polish = optimize_weights(best_h, DEADLINE - 2 - elapsed(), start)
        if val_polish > best_val:
            best_val = val_polish
            best_h = h_polish.copy()
            
    # Ensure non-negative and return as list
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
# EVOLVE-BLOCK-END
