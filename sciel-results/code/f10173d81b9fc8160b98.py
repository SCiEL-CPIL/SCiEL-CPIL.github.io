import time
import random
import numpy as np

def construct():
    N = 51975
    # Find all divisors > 1
    divisors = []
    for i in range(2, N + 1):
        if N % i == 0:
            divisors.append(i)
    
    # Sort divisors? Maybe smaller first for initialization?
    # Doesn't matter much for SA.
    
    # Initialize residues
    # Greedy init: for each m, pick r that covers most uncovered.
    # But we don't have 'uncovered' yet.
    # Just pick random r? Or r=0?
    # Let's do a quick greedy init.
    covered = np.zeros(N, dtype=np.uint8)
    residues = {}
    # Sort by m ascending? Smaller m have more flexibility?
    # Actually, let's just pick random residues initially.
    for m in divisors:
        residues[m] = random.randint(0, m - 1)
        
    # Calculate initial coverage
    # Using boolean array is faster for XOR/OR logic?
    # But we need counts for SA.
    counts = np.zeros(N, dtype=np.int32)
    for m in divisors:
        r = residues[m]
        counts[r::m] += 1
        
    uncovered = np.sum(counts == 0)
    best_uncovered = uncovered
    best_residues = residues.copy()
    
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    T = 10.0 # Initial temperature
    cooling = 0.999
    
    # Precompute ranges?
    # For each m, we need to access counts[r::m]
    
    while time.time() - start_time < timeout:
        # Pick a random modulus
        m = random.choice(divisors)
        r_old = residues[m]
        
        # Find best r for this m
        best_r = r_old
        best_delta = 0 # delta for staying is 0
        
        # We can sample a few r instead of all if m is large?
        # m is at most 51975. But most are small.
        # For m=3, 3 options. For m=51975, 51975 options.
        # If m is large, N/m is small (1 or 2).
        # Cost of checking all r is sum_{r=0}^{m-1} (N/m) = N.
        # So checking all r for any m costs O(N).
        # This is 52k ops.
        # 47 moduli. Total 2.4M ops per full sweep.
        # We are doing 1 move per iteration.
        # So each iteration is O(N).
        # 120s / (52k ops / 10^7 ops/s) ~ 120 / 0.005 = 24000 iterations.
        # This is fine.
        
        # Optimization: Only check r if it's different?
        # We want the best r.
        # Let's just iterate all r.
        for r_new in range(m):
            if r_new == r_old:
                continue
            
            # Calculate delta
            # Old set S_old, New set S_new
            # Delta U = (count of 1s in S_old) - (count of 0s in S_new)
            # Note: S_old and S_new are disjoint.
            
            # We need to be careful with the counts array.
            # Currently counts reflects the state with r_old.
            
            c_old = counts[r_old::m]
            c_new = counts[r_new::m]
            
            # If we switch to r_new:
            # Elements in S_old will have count decreased by 1.
            # Elements in S_new will have count increased by 1.
            
            # Uncovered increases by number of x in S_old where count[x] == 1.
            inc = np.sum(c_old == 1)
            # Uncovered decreases by number of x in S_new where count[x] == 0.
            dec = np.sum(c_new == 0)
            
            delta = inc - dec
            
            if delta < best_delta:
                best_delta = delta
                best_r = r_new
            elif delta == best_delta:
                # Tie-breaking: random?
                if random.random() < 0.5:
                    best_r = r_new
        
        # Decide whether to move to best_r
        if best_delta < 0:
            # Always accept improvement
            accept = True
        elif best_delta == 0:
            # Accept sideways moves to explore
            accept = random.random() < 0.5
        else:
            # Accept worse moves with prob exp(-delta/T)
            accept = random.random() < np.exp(-best_delta / T)
            
        if accept:
            # Apply move
            residues[m] = best_r
            # Update counts
            counts[r_old::m] -= 1
            counts[best_r::m] += 1
            uncovered += best_delta
            
            if uncovered < best_uncovered:
                best_uncovered = uncovered
                best_residues = residues.copy()
                
        # Cool down
        T *= cooling
        if T < 0.1:
            T = 0.1
            
    # Return best found
    # Format: [[m, r], ...]
    result = []
    for m in divisors:
        result.append([m, best_residues[m]])
        
    return result
