import numpy as np
import time
import random

def construct():
    N = 51975
    divisors = [d for d in range(2, N + 1) if N % d == 0]
    divisors_asc = sorted(divisors)
    divisors_desc = sorted(divisors, reverse=True)
    
    # Precompute x % d for all d
    x_mods = {}
    for d in divisors:
        x_mods[d] = np.arange(N) % d
    
    def greedy_build(order):
        residues = {}
        count = np.zeros(N, dtype=np.int8)
        for d in order:
            # Find r that maximizes newly covered residues
            # For each r, count how many x in range(r, N, d) have count[x] == 0
            best_r = 0
            best_gain = -1
            # Use bincount on uncovered residues
            uncovered = (count == 0)
            # For each r, gain = sum(uncovered[r::d])
            # Vectorize: for each r, we want sum of uncovered at positions r, r+d, r+2d, ...
            # This is equivalent to: take uncovered array, reshape to (-1, d) won't work directly
            # But we can do: for each r, sum uncovered[r::d]
            # Faster: use the x_mods approach
            # gain[r] = number of uncovered x where x % d == r
            # But we also need to only count x that are actually in the arithmetic progression
            # x in range(r, N, d) means x ≡ r (mod d) and x >= r
            # Since N is a multiple of d, the number of such x is exactly N/d
            # And they are exactly {x in [0, N) : x % d == r}
            # So gain[r] = sum of uncovered[x] for x where x % d == r
            mods = x_mods[d][uncovered]
            gains = np.bincount(mods, minlength=d)
            best_r = int(np.argmax(gains))
            best_gain = gains[best_r]
            residues[d] = best_r
            count[best_r::d] += 1
        uncovered_count = np.sum(count == 0)
        return residues, count, uncovered_count
    
    # Try both ascending and descending greedy, pick the better
    res_asc, cnt_asc, unc_asc = greedy_build(divisors_asc)
    res_desc, cnt_desc, unc_desc = greedy_build(divisors_desc)
    
    if unc_asc <= unc_desc:
        residues = res_asc
        count = cnt_asc.copy()
    else:
        residues = res_desc
        count = cnt_desc.copy()
    
    start_time = time.time()
    timeout = 110.0
    
    while time.time() - start_time < timeout:
        improved = False
        shuffled_divs = divisors[:]
        random.shuffle(shuffled_divs)
        
        for d in shuffled_divs:
            if time.time() - start_time > timeout:
                break
                
            r_old = residues[d]
            
            # Get indices of uncovered residues
            uncovered_idx = np.nonzero(count == 0)[0]
            
            if len(uncovered_idx) == 0:
                return [[d, int(residues[d])] for d in divisors]
                
            # Compute gain for each r: number of uncovered x with x % d == r
            mods = x_mods[d][uncovered_idx]
            gain = np.bincount(mods, minlength=d)
            
            # For each candidate r_new, compute delta:
            # delta(r_new) = gain[r_new] - (number of x in range(r_new, N, d) with count[x] == 1)
            # But when we remove r_old, some x in range(r_old, N, d) become count 0
            # and some become count 1 (if they were count 2)
            # This is complex. Simpler: just evaluate the actual change for the best candidate.
            
            # Find best r by gain
            best_r = int(np.argmax(gain))
            
            if best_r != r_old:
                # Evaluate the actual delta by simulating
                # Old: count[r_old::d] contributes
                # New: count[best_r::d] contributes
                # delta = (newly uncovered) - (newly covered)
                # Newly covered: x in range(best_r, N, d) where count[x] == 0
                # Newly uncovered: x in range(r_old, N, d) where count[x] == 1
                newly_covered = np.sum(count[best_r::d] == 0)
                newly_uncovered = np.sum(count[r_old::d] == 1)
                
                if newly_covered > newly_uncovered:
                    count[r_old::d] -= 1
                    count[best_r::d] += 1
                    residues[d] = best_r
                    improved = True
        
        if time.time() - start_time > timeout:
            break
            
        if not improved:
            # Try random perturbations
            for _ in range(20):
                if time.time() - start_time > timeout:
                    break
                d = random.choice(divisors)
                r_old = residues[d]
                r_new = random.randint(0, d - 1)
                if r_new == r_old:
                    continue
                
                newly_covered = np.sum(count[r_new::d] == 0)
                newly_uncovered = np.sum(count[r_old::d] == 1)
                
                if newly_covered > newly_uncovered:
                    count[r_old::d] -= 1
                    count[r_new::d] += 1
                    residues[d] = r_new
                    improved = True
                else:
                    break
                    
            if not improved:
                break

    return [[d, int(residues[d])] for d in divisors]
