import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 50

def compute_score(h):
    """
    Computes the ratio: ||h||_2^2 / (||h||_1 * ||h * h||_infty)
    """
    if np.sum(h) == 0:
        return 0.0
    l1 = np.sum(h)
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    l_inf = np.max(conv)
    if l_inf == 0:
        return 0.0
    return l2_sq / (l1 * l_inf)

def find_sidon_set(n, target_size=10):
    """
    Greedily finds a Sidon set of size target_size in [0, n).
    """
    S = [0]
    used_sums = set([0])
    while len(S) < target_size:
        found = False
        for i in range(1, n):
            if i in S:
                continue
            # Check if adding i creates duplicate sums
            valid = True
            new_sums = set()
            for j in S:
                s = i + j
                if s in used_sums:
                    valid = False
                    break
                new_sums.add(s)
            if valid:
                S.append(i)
                used_sums.update(new_sums)
                found = True
                break
        if not found:
            break
    return S

def optimize_weights(support, time_limit=10):
    """
    Optimizes weights on the given support to maximize the score.
    """
    K = len(support)
    # Initial guess: equal weights
    h0 = np.ones(K) / K
    
    def objective(w):
        # w are the weights on the support
        # Normalize to L1=1
        w_norm = w / np.sum(w)
        h_full = np.zeros(N)
        h_full[support] = w_norm
        return -compute_score(h_full)

    # Use Nelder-Mead for non-smooth objective
    # We might need multiple restarts
    best_score = 0.0
    best_w = h0
    
    # Run a few iterations of Nelder-Mead
    # To save time, we limit the number of function evaluations
    # Nelder-Mead is local, so we do a few random restarts
    
    start_time = time.time()
    for restart in range(3):
        if restart == 0:
            w_start = h0
        else:
            w_start = np.random.dirichlet(np.ones(K))
            
        res = minimize(objective, w_start, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.fun < -best_score:
            best_score = -res.fun
            best_w = res.x
            
        if time.time() - start_time > time_limit:
            break
            
    h_full = np.zeros(N)
    h_full[support] = best_w / np.sum(best_w)
    return h_full, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_h = np.ones(N) / N
    best_score = compute_score(best_h)
    
    # Strategy 1: Try several Sidon sets of increasing size
    # Max Sidon size for N=50 is around 7-8.
    # We'll try sizes 5, 6, 7, 8.
    
    # Generate a few different Sidon sets using different seeds/strategies
    # Since find_sidon_set is deterministic, we might get the same set.
    # To get variety, we can shuffle the search order or use different starting points.
    
    candidates = []
    
    # Standard greedy Sidon
    S1 = find_sidon_set(N, target_size=8)
    candidates.append(S1)
    
    # Try to find another one by skipping 0? Or different greedy order?
    # Let's just try a random sparse set as a fallback if Sidon fails or is slow.
    
    # Optimization on Sidon set
    if time.time() < deadline:
        h_opt, score_opt = optimize_weights(S1, time_limit=30)
        if score_opt > best_score:
            best_score = score_opt
            best_h = h_opt
    
    # Strategy 2: Try a "Golomb Ruler" like structure or random sparse sets
    # Random sparse sets might not be Sidon, but if K is small, collisions are few.
    # If K=5, collisions are manageable.
    
    for _ in range(5):
        if time.time() > deadline - 20:
            break
            
        # Pick 6 random indices
        support = np.random.choice(N, size=6, replace=False)
        support = sorted(support)
        
        h_opt, score_opt = optimize_weights(support, time_limit=10)
        if score_opt > best_score:
            best_score = score_opt
            best_h = h_opt
            
    # Strategy 3: If time remains, try a global random search with local refinement
    # This is expensive, so only if we have > 30s left.
    
    if time.time() < deadline - 30:
        # Try a few random dense-ish vectors? No, sparse is better.
        # Try optimizing a specific pattern: alternating?
        # Or just more random sparse supports.
        for _ in range(10):
            if time.time() > deadline - 10:
                break
            K = np.random.randint(4, 9)
            support = np.random.choice(N, size=K, replace=False)
            support = sorted(support)
            h_opt, score_opt = optimize_weights(support, time_limit=5)
            if score_opt > best_score:
                best_score = score_opt
                best_h = h_opt

    return list(best_h)
