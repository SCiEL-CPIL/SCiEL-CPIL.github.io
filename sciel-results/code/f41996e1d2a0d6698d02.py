import time
import random
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin for safety
    
    n = 30
    # Calculate LCM of 1..30
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # Weights w[i] = L / (i+1) for i in 0..29
    # c[i] corresponds to 1/(i+1)
    w = [L // (i + 1) for i in range(n)]
    
    best_val = None
    best_coeffs = None
    
    # Initial seed: [0]*28 + [1, -1]
    # Sum = 1/29 - 1/30 = 1/870
    # N_seed = L/870
    seed_coeffs = [0] * 28 + [1, -1]
    S_seed = sum(c * w[i] for i, c in enumerate(seed_coeffs))
    if S_seed != 0:
        best_val = abs(S_seed)
        best_coeffs = seed_coeffs[:]
    else:
        best_val = float('inf')
        best_coeffs = [0] * n

    # Helper to compute sum
    def calc_sum(coeffs):
        return sum(c * w[i] for i, c in enumerate(coeffs))
    
    # Local search function
    def local_search(curr, curr_sum, curr_abs, time_limit):
        """
        Perform local search starting from curr.
        Returns (best_coeffs, best_val) found during this search.
        """
        local_best_coeffs = curr[:]
        local_best_val = curr_abs
        
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < time_limit:
            improved = False
            
            # Try all single flips (steepest descent / first improvement)
            # We iterate through indices and try changing to -1, 0, 1
            for i in range(n):
                if time.time() > time_limit:
                    break
                old_c = curr[i]
                
                for new_c in [-1, 0, 1]:
                    if new_c == old_c:
                        continue
                    
                    delta = (new_c - old_c) * w[i]
                    new_sum = curr_sum + delta
                    new_abs = abs(new_sum) if new_sum != 0 else float('inf')
                    
                    # Accept if new_abs is better (smaller) than curr_abs
                    # Note: if curr_abs is inf (sum was 0), any finite new_abs is better
                    if new_abs < curr_abs:
                        curr[i] = new_c
                        curr_sum = new_sum
                        curr_abs = new_abs
                        improved = True
                        
                        # Update local best
                        if curr_abs < local_best_val:
                            local_best_val = curr_abs
                            local_best_coeffs = curr[:]
                        
                        # Restart the scan for this index? Or break to outer?
                        # Breaking to outer allows re-scanning from index 0, which is more thorough.
                        # But it's slower. Let's break to outer for correctness of "hill climbing".
                        break
                if improved:
                    break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    break
            else:
                no_improve_count = 0
        
        return local_best_coeffs, local_best_val

    random.seed(42)
    
    # Phase 1: Local search from the seed
    curr = best_coeffs[:]
    curr_sum = calc_sum(curr)
    curr_abs = abs(curr_sum) if curr_sum != 0 else float('inf')
    
    time_budget_1 = min(10.0, deadline - 5.0 - start_time)
    if time_budget_1 > 1.0:
        found_coeffs, found_val = local_search(curr, curr_sum, curr_abs, start_time + time_budget_1)
        if found_val < best_val:
            best_val = found_val
            best_coeffs = found_coeffs
    
    # Phase 2: Random restarts with local search
    # Use the best found so far as a reference for "good"
    # But random restarts explore different basins.
    
    max_restarts = 1000
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random initialization
        # Bias towards sparse solutions? Or uniform?
        # The seed is sparse. Let's try a mix.
        # 50% chance: start from best_coeffs and flip a few
        # 50% chance: fully random
        if restart % 2 == 0 and best_coeffs is not None:
            curr = best_coeffs[:]
            # Flip 2-5 random indices
            num_flips = random.randint(2, 5)
            for _ in range(num_flips):
                idx = random.randint(0, n - 1)
                curr[idx] = random.choice([-1, 0, 1])
        else:
            # Sparse random: pick 3-10 indices to be non-zero
            num_nonzero = random.randint(3, 10)
            curr = [0] * n
            positions = random.sample(range(n), num_nonzero)
            for pos in positions:
                curr[pos] = random.choice([-1, 1])
        
        curr_sum = calc_sum(curr)
        curr_abs = abs(curr_sum) if curr_sum != 0 else float('inf')
        
        # If we started with inf, we must find a non-zero sum
        # Local search handles this by accepting any finite improvement over inf
        
        # Time budget for this restart
        # Decrease budget as we go? Or fixed?
        # Let's use a fixed small budget per restart, e.g., 0.5s
        # But we have ~60s total. 1000 restarts * 0.06s = 60s.
        # Let's aim for ~50 restarts in the remaining time if each takes ~1s?
        # Or many fast ones?
        # Local search can be fast if it converges quickly.
        
        time_budget = 0.5  # seconds per restart
        found_coeffs, found_val = local_search(curr, curr_sum, curr_abs, start_time + time_budget)
        
        if found_val < best_val:
            best_val = found_val
            best_coeffs = found_coeffs
            
            # If we found a very good value, we might want to do more local search from it
            # But for now, let's just record it.
            
    return best_coeffs
# EVOLVE-BLOCK-END
