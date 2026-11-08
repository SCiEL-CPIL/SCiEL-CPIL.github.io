"""
Strategy: Exhaustive search over small sets using optimized score computation.
The score function S(U) = 1 + log(|U-U|/|U+U|) / log(2*max(U)+1) is maximized by sets
with high "difference-to-sum" ratio relative to their range.
We combine the exhaustive search structure of the current program with the
vectorized numpy score computation from the inspiration for speed.
"""
import time
import math
import numpy as np
from itertools import combinations

def compute_score(U):
    """
    Computes the lower bound exponent: 1 + log(|U-U|/|U+U|)/log(2*max(U) + 1)
    Uses vectorized numpy for speed.
    """
    if len(U) < 2:
        return 0.0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    max_sum = 2 * maxU
    n_fft = 1
    while n_fft < max_sum + 1:
        n_fft *= 2
        
    A = np.zeros(n_fft, dtype=np.float64)
    for u in U:
        A[u] = 1.0
        
    A_fft = np.fft.rfft(A)
    C_fft = A_fft * A_fft
    C = np.fft.irfft(C_fft, n=n_fft)
    sum_set_size = np.sum(C[:max_sum + 1] > 0.5)
    
    B = np.zeros(n_fft, dtype=np.float64)
    for u in U:
        B[maxU - u] = 1.0
        
    B_fft = np.fft.rfft(B)
    D_fft = A_fft * B_fft
    D = np.fft.irfft(D_fft, n=n_fft)
    diff_set_size = np.sum(D[:max_sum + 1] > 0.5)
    
    if sum_set_size == 0:
        return 0.0
        
    log_ratio = math.log(diff_set_size / sum_set_size)
    log_range = math.log(2 * maxU + 1)
    
    return 1.0 + log_ratio / log_range

def construct():
    """
    Entry point. Returns a list of nonnegative integers containing 0.
    Uses exhaustive search for small sets to find the global optimum.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s buffer for 150s limit
    
    best_score = -1.0
    best_U = [0, 1]
    
    # Search for sets of size N containing 0.
    # We choose N-1 elements from {1, ..., K}.
    # Since score decreases with max(U) for a fixed ratio, we prioritize small K.
    
    max_N = 10
    max_K = 30
    
    # We will iterate over N and K.
    # For each (N, K), we generate all combinations of size N-1 from {1, ..., K}.
    
    # To manage time, we estimate the number of combinations and skip if too large.
    # C(K, N-1) should be manageable.
    
    for N in range(3, max_N + 1):
        if time.time() > deadline:
            break
            
        # Determine max K for this N to keep C(K, N-1) reasonable
        # C(30, 9) is huge. C(20, 7) is 77k.
        # Let's be conservative.
        # For N=3, C(K, 2) ~ K^2/2. K=100 -> 5k. OK.
        # For N=4, C(K, 3) ~ K^3/6. K=50 -> 20k. OK.
        # For N=5, C(K, 4) ~ K^4/24. K=30 -> 28k. OK.
        # For N=6, C(K, 5) ~ K^5/120. K=25 -> 53k. OK.
        # For N=7, C(K, 6) ~ K^6/720. K=20 -> 39k. OK.
        # For N=8, C(K, 7) ~ K^7/5040. K=15 -> 64k. OK.
        # For N=9, C(K, 8) ~ K^8/40320. K=12 -> 49k. OK.
        # For N=10, C(K, 9) ~ K^9/362880. K=10 -> 10k. OK.
        
        # Let's set max_K based on N
        if N == 3:
            max_K_for_N = 150
        elif N == 4:
            max_K_for_N = 80
        elif N == 5:
            max_K_for_N = 50
        elif N == 6:
            max_K_for_N = 35
        elif N == 7:
            max_K_for_N = 25
        elif N == 8:
            max_K_for_N = 20
        elif N == 9:
            max_K_for_N = 15
        else:
            max_K_for_N = 12
            
        for K in range(N, max_K_for_N + 1):
            if time.time() > deadline - 5:
                break
                
            # Estimate combinations
            try:
                count = math.comb(K, N-1)
            except:
                count = 10**9
                
            if count > 500000:
                continue
                
            # Generate combinations of size N-1 from {1, ..., K}
            # 0 is always in the set.
            # We want to include K? Not necessarily.
            # The max element of U will be the max of the combination.
            # So iterating K as the max element is one way, but we can just
            # iterate all subsets of {1..K} of size N-1.
            
            # To be efficient, we can just generate all combinations.
            # itertools.combinations is fast in C.
            
            for comb in combinations(range(1, K + 1), N - 1):
                U = [0] + list(comb)
                s = compute_score(U)
                if s > best_score:
                    best_score = s
                    best_U = U
                    
    # Refine: Local search on the best found
    current_U = list(best_U)
    current_score = best_score
    
    # Simple local search to polish
    while time.time() < deadline - 2:
        improved = False
        current_max = max(current_U) if current_U else 0
        
        # Try adding an element
        for x in range(1, current_max + 10):
            if x not in current_U:
                new_U = sorted(list(set(current_U + [x])))
                s = compute_score(new_U)
                if s > current_score:
                    current_U = new_U
                    current_score = s
                    improved = True
                    break
        
        # Try removing an element
        for x in current_U:
            if x == 0:
                continue
            new_U = [y for y in current_U if y != x]
            if len(new_U) < 2:
                continue
            s = compute_score(new_U)
            if s > current_score:
                current_U = new_U
                current_score = s
                improved = True
                break
        
        # Try swapping
        if not improved:
             for i, x in enumerate(current_U):
                 if x == 0: continue
                 for y in range(1, current_max + 5):
                     if y == x or y in current_U: continue
                     new_U = list(current_U)
                     new_U[i] = y
                     new_U = sorted(list(set(new_U)))
                     s = compute_score(new_U)
                     if s > current_score:
                         current_U = new_U
                         current_score = s
                         improved = True
                         break
                 if improved: break
                 
        if not improved:
            break
            
    if current_score > best_score:
        best_U = current_U
        best_score = current_score
        
    return best_U
