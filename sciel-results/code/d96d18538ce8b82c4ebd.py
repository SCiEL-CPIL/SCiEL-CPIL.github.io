import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50

def find_sidon_set(n, max_size=10, num_trials=1000):
    """
    Finds a Sidon subset of {0, ..., n-1} of size max_size.
    A Sidon set is one where all pairwise sums are unique.
    """
    for size in range(max_size, 2, -1):
        for _ in range(num_trials):
            S = set(random.sample(range(n), size))
            S_list = list(S)
            sums = set()
            is_sidon = True
            for i in range(size):
                for j in range(i, size):
                    s = S_list[i] + S_list[j]
                    if s in sums:
                        is_sidon = False
                        break
                    sums.add(s)
                if not is_sidon:
                    break
            if is_sidon:
                return S_list
    # Fallback to a small known set or greedy
    # Greedy construction
    S = [0]
    for x in range(1, n):
        # Check if x can be added
        ok = True
        for y in S:
            if x + y in [a + b for a in S for b in S if a <= b]:
                 # This check is inefficient, need to store sums
                 pass
        # Simpler fallback: just return first 'size' elements of a known sequence or uniform
        # For robustness, if random fails, use a heuristic
        # Let's just try a few more random trials with smaller size
    for size in range(max_size, 1, -1):
        for _ in range(5000):
            S = set(random.sample(range(n), size))
            S_list = list(S)
            sums = set()
            is_sidon = True
            for i in range(size):
                for j in range(i, size):
                    s = S_list[i] + S_list[j]
                    if s in sums:
                        is_sidon = False
                        break
                    sums.add(s)
                if not is_sidon:
                    break
            if is_sidon:
                return S_list
    # Ultimate fallback: uniform
    return list(range(n))

def _smooth_ratio(h, beta):
    """
    Compute the smooth approximation of the ratio:
    ||h||_2^2 / (||h||_1 * max(conv(h)))
    using log-sum-exp for the max.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-10):
        return -1e10 # Penalty for invalid
    h = np.maximum(h, 0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return -1e10
    l2_sq = np.sum(h * h)
    
    conv = np.convolve(h, h, mode='full')
    
    # Log-sum-exp trick for numerical stability
    # max(c) ~ (1/beta) * log(sum(exp(beta * c)))
    # We want to maximize l2_sq / (l1 * max_c)
    # This is equivalent to minimizing -l2_sq / (l1 * max_c)
    # Or maximizing log(l2_sq) - log(l1) - log(max_c)
    # log(max_c) ~ (1/beta) * log(sum(exp(beta * c))) - (1/beta) * beta * max_c ? No.
    # log(sum(exp(beta*c))) = beta * max_c + log(sum(exp(beta*(c-max_c))))
    # So (1/beta) * log(sum(exp(beta*c))) = max_c + (1/beta) * log(sum(exp(beta*(c-max_c))))
    # This is a smooth upper bound on max_c.
    
    c_max = np.max(conv)
    if c_max < 1e-15:
        return -1e10
    
    # Shift to prevent overflow
    conv_shifted = conv - c_max
    # log_sum_exp = log(sum(exp(beta * conv_shifted)))
    # We need to handle beta * conv_shifted
    # If beta is large, this might underflow for most terms, which is fine.
    # sum_exp = np.sum(np.exp(beta * conv_shifted))
    # log_max_approx = c_max + (1/beta) * np.log(sum_exp)
    
    # To avoid overflow in exp, we can just use the shifted version.
    # Note: conv_shifted <= 0. So exp(beta * conv_shifted) <= 1.
    # Sum is at least 1 (for the max term).
    
    sum_exp = np.sum(np.exp(beta * conv_shifted))
    log_max_approx = c_max + (1.0 / beta) * np.log(sum_exp)
    
    if log_max_approx < 1e-15:
        return -1e10
        
    ratio = l2_sq / (l1 * log_max_approx)
    return ratio

def optimize_weights(indices, time_limit=100):
    """
    Optimizes weights on the given indices to maximize the ratio.
    Uses scipy.optimize.minimize with a smooth surrogate.
    """
    n = N
    M = len(indices)
    if M == 0:
        return np.zeros(n)
    
    # Initial guess: uniform on indices
    h0 = np.zeros(n)
    for idx in indices:
        h0[idx] = 1.0 / M
        
    start_time = time.time()
    best_h = h0.copy()
    best_score = _smooth_ratio(h0, beta=10.0)
    
    # Try different beta values (annealing from smooth to sharp)
    betas = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    def objective(x):
        # x is the full vector h
        return -_smooth_ratio(x, beta)
        
    # Constraints: sum(h) = 1, h >= 0
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    bounds = [(0.0, 1.0) for _ in range(n)]
    
    for beta in betas:
        if time.time() - start_time > time_limit - 5:
            break
            
        # Re-initialize from best_h found so far to preserve structure?
        # Or start from h0? Starting from best_h is better for refinement.
        h_current = best_h.copy()
        
        try:
            res = minimize(objective, h_current, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-12})
            if res.success or res.fun < -best_score:
                h_new = res.x
                # Ensure non-negativity and normalization
                h_new = np.maximum(h_new, 0)
                s = np.sum(h_new)
                if s > 1e-15:
                    h_new /= s
                score_new = _smooth_ratio(h_new, beta)
                if score_new > best_score:
                    best_score = score_new
                    best_h = h_new
        except Exception:
            pass

    return best_h

def construct():
    start_time = time.time()
    deadline = 140 # Leave 10s margin for safety (kill is 150s)
    
    # Find a good Sidon set
    # Try sizes 8, 7, 6...
    indices = None
    for size in range(8, 3, -1):
        if time.time() - start_time > deadline - 20:
            break
        indices = find_sidon_set(N, max_size=size, num_trials=2000)
        if len(indices) == size:
            break
            
    if indices is None or len(indices) < 2:
        # Fallback to uniform
        return [1.0/N] * N
        
    # Optimize weights
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time < 5:
        # Not enough time, just return uniform on Sidon set
        h = np.zeros(N)
        for idx in indices:
            h[idx] = 1.0 / len(indices)
        return h.tolist()
        
    h = optimize_weights(indices, time_limit=remaining_time)
    
    # Ensure non-negative and normalized
    h = np.maximum(h, 0)
    s = np.sum(h)
    if s > 0:
        h = h / s
    else:
        h = np.ones(N) / N
        
    return h.tolist()
