import math
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s kill
    
    N = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, N + 1):
        L = L * i // gcd(L, i)
        
    # Precompute weights w_i = L // i
    # weights[0] corresponds to i=1, ..., weights[29] to i=30
    weights = [L // i for i in range(1, N + 1)]

    def calc_sum(coeffs):
        s = 0
        for i in range(N):
            if coeffs[i] != 0:
                s += coeffs[i] * weights[i]
        return s

    best_abs_num = float('inf')
    best_coeffs = [1] + [0] * (N - 1) # Fallback: sum = 1

    # Strategy:
    # 1. Random initialization + Hill Climbing (Inspiration 2/3)
    # 2. If we get stuck or finish, we can try to improve with more sophisticated local search.
    # The current program (meet-in-the-middle) is too slow for 3^15 in Python without C extensions.
    # However, the "inspiration 1" approach (enumerating small support sets) is very strong for finding
    # small numerators because it restricts the search space significantly.
    # Let's combine:
    # Phase 1: Quick random search to find a decent baseline.
    # Phase 2: Systematic search over small support sets (k=2,3,4,5) to find the global minimum
    #          among sparse solutions. This is often the case for harmonic sums because
    #          cancellations between large terms are hard to achieve with many terms due to size mismatch,
    #          but small subsets can cancel exactly or near-exactly.
    
    # Actually, the meet-in-the-middle in the current program tries to find ANY solution with numerator 1.
    # If it fails (or is too slow), we fall back.
    # The inspirations show that sparse solutions (small k) can yield very small numerators.
    # Score 10852 in inspiration 1 suggests that finding a small numerator via sparse search is effective.
    # Score 238 in inspiration 2 suggests random search is okay but not great.
    # Score 23 in inspiration 3 suggests local search from random start is decent.
    
    # Let's prioritize the sparse search (Inspiration 1) because it's deterministic and finds
    # provably small numerators for small k. Then use local search to improve if time permits.
    
    # Phase 1: Sparse Search (from Inspiration 1)
    # We look for subsets of indices of size k, and sign assignments, such that |sum| is minimized and non-zero.
    # k=2: C(30,2) * 4 = 435 * 4 = 1740 checks. Fast.
    # k=3: C(30,3) * 8 = 4060 * 8 = 32480 checks. Fast.
    # k=4: C(30,4) * 16 = 27405 * 16 = 438480 checks. ~0.5s in Python.
    # k=5: C(30,5) * 32 = 142506 * 32 = 4.5M checks. ~5-10s in Python.
    # k=6: C(30,6) * 64 = 593775 * 64 = 38M checks. Too slow in pure Python.
    
    # We will run k=2,3,4 fully. For k=5, we might sample or use a smarter method if time allows.
    # But even k=4 often finds numerator 1 or very small.
    
    from itertools import combinations
    
    max_k = 4
    # For k=5, we can try a limited number of random subsets if time permits, or just skip if k<=4 finds 1.
    
    for k in range(2, max_k + 1):
        if time.time() > deadline - 10:
            break
            
        # Iterate over all combinations of k indices
        for combo in combinations(range(N), k):
            # Precompute weights for this combo
            w_combo = [weights[i] for i in combo]
            
            # We want to assign signs s_i in {-1, 1} to minimize |sum s_i w_i| != 0
            # Note: Coefficients can be 0, but in this sparse search we only use -1 and 1 for the selected indices.
            # This is a subset of the full search space.
            
            # Meet-in-the-middle for the subset?
            # k is small (<=4), so 2^k is small.
            # Just enumerate all 2^k sign patterns.
            
            # To speed up, we can sort the weights? No, indices are fixed.
            # Just iterate masks.
            
            for mask in range(1 << k):
                s = 0
                # Calculate sum
                # mask bit i is 1 -> sign -1? or +1?
                # Let's say bit i is 1 -> -1, 0 -> +1
                # Sum = sum(w_i) - 2 * sum(w_i for i where bit is 1)
                
                # Optimization: Calculate total_pos once?
                # No, we can just sum directly for small k.
                # Or use the total_pos - 2*subset_sum trick.
                
                # Let's use the direct sum for clarity, it's fast for k<=4.
                for j in range(k):
                    if (mask >> j) & 1:
                        s -= w_combo[j]
                    else:
                        s += w_combo[j]
                
                if s != 0:
                    abs_s = abs(s)
                    if abs_s < best_abs_num:
                        best_abs_num = abs_s
                        # Construct coefficients
                        coeffs = [0] * N
                        for j in range(k):
                            idx = combo[j]
                            if (mask >> j) & 1:
                                coeffs[idx] = -1
                            else:
                                coeffs[idx] = 1
                        best_coeffs = coeffs
                        
                        if best_abs_num == 1:
                            # Can't get better than 1
                            return best_coeffs

    # Phase 2: Local Search to improve if we didn't find 1
    # Start from best_coeffs found in Phase 1, or random if Phase 1 failed (unlikely).
    # We can also try random restarts with local search.
    
    # Let's refine the best found so far using local search.
    current_coeffs = best_coeffs[:]
    current_sum = calc_sum(current_coeffs)
    current_abs = abs(current_sum)
    
    # Also try a few random starts with local search to see if we can beat the sparse solution
    # The sparse solution might be a local minimum for dense moves.
    
    random.seed(42)
    
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Start from a random configuration or the current best
        if r == 0:
            c = best_coeffs[:]
        else:
            c = [random.choice([-1, 0, 1]) for _ in range(N)]
        
        s = calc_sum(c)
        if s == 0:
            # Perturb to move away from 0
            i = random.randint(0, N-1)
            c[i] = 1 if c[i] != 1 else -1
            s = calc_sum(c)
            if s == 0:
                continue
            
        abs_s = abs(s)
        
        # Local Search: Single-bit flips
        improved = True
        steps = 0
        max_steps = 1000
        while improved and steps < max_steps and time.time() < deadline - 2:
            improved = False
            steps += 1
            # Try flipping each coefficient
            # To speed up, we can iterate in random order or systematic.
            # Systematic is fine for N=30.
            for i in range(N):
                old_val = c[i]
                # Try changing to other two values
                # Values are -1, 0, 1
                # If old is -1, try 0, 1
                # If old is 0, try -1, 1
                # If old is 1, try -1, 0
                
                candidates = []
                if old_val == -1:
                    candidates = [0, 1]
                elif old_val == 0:
                    candidates = [-1, 1]
                else:
                    candidates = [-1, 0]
                    
                for new_val in candidates:
                    # Calculate delta
                    # s_new = s + (new_val - old_val) * weights[i]
                    delta = (new_val - old_val) * weights[i]
                    s_new = s + delta
                    
                    if s_new != 0:
                        abs_new = abs(s_new)
                        if abs_new < abs_s:
                            # Accept
                            c[i] = new_val
                            s = s_new
                            abs_s = abs_new
                            improved = True
                            
                            if abs_s == 1:
                                best_abs_num = 1
                                best_coeffs = c[:]
                                return best_coeffs
                            
                            # Break out of candidate loop, continue to next i?
                            # Or restart the scan?
                            # First improvement strategy: break and restart scan.
                            break
                # If we improved, we can break out of the i loop and restart the while loop
                # to check for more improvements from the new state.
                if improved:
                    break
        
        if abs_s < best_abs_num:
            best_abs_num = abs_s
            best_coeffs = c[:]
            if best_abs_num == 1:
                return best_coeffs

    return best_coeffs
