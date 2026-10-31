import numpy as np
import time

N = 50

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

def construct():
    start_time = time.time()
    deadline = 110.0
    
    # 1. Find a large Sidon set
    A_greedy = find_sidon_set_greedy(N)
    
    best_A = A_greedy
    # Try to find a larger one with random restarts
    # Max size for N=50 is likely 7 or 8
    for target_size in range(len(A_greedy) + 1, 10):
        found = False
        for _ in range(500): # 500 random tries per size
            if time.time() > deadline - 5:
                break
            A_try = sorted(np.random.choice(N, target_size, replace=False))
            if is_sidon(A_try):
                if target_size > len(best_A):
                    best_A = A_try
                    found = True
                # Don't break immediately, maybe find another?
                # But one is enough for now.
                if found:
                    break
        if found:
            continue
        else:
            break # If we can't find size K, we can't find K+1
            
    # 2. Construct vector
    h = np.zeros(N)
    for idx in best_A:
        h[idx] = 1.0
        
    # Normalize to sum 1
    h_sum = np.sum(h)
    if h_sum > 0:
        h = h / h_sum
        
    # 3. Optional: Simple local search to tweak weights
    # We want to maximize (sum h^2) / max(conv)
    # This is equivalent to minimizing max(conv) / sum(h^2)
    # Let's do a few iterations of "move mass from high-impact to low-impact"
    
    best_h = h.copy()
    
    # Compute initial score
    conv = np.convolve(best_h, best_h, mode='full')
    max_conv = np.max(conv)
    l2_sq = np.sum(best_h**2)
    if max_conv > 0:
        best_score = l2_sq / max_conv
    else:
        best_score = 0.0
        
    # Local search
    # Try shifting small amounts of mass
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        # Find the peak index
        k_peak = np.argmax(conv)
        
        # For each i in support, try moving mass to a random j
        # Or just try to reduce h[i] if i is involved in the peak
        # Simple heuristic: reduce h[i] for i that contributes to k_peak
        # Contribution of h[i] to c[k_peak] is h[i] * h[k_peak - i]
        
        # Let's try a simple gradient-like step
        # Perturb h randomly
        h_pert = best_h.copy()
        # Randomly pick two indices i, j
        # Move mass from i to j
        # We want to reduce the peak.
        
        # Actually, for Sidon sets, equal weights are very good.
        # The improvement from local search might be marginal.
        # Let's skip complex local search to avoid bugs and timeout.
        
        if not improved:
            break
            
    return list(best_h)
