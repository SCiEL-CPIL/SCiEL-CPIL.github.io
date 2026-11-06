import time
import numpy as np
import random

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    N = 51975
    
    # 1. Generate divisors
    divisors = []
    for i in range(2, int(np.sqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    divisors.append(N)
    divisors = list(set(divisors))
    divisors.sort()
    
    # 2. Precompute indices for each modulus and residue
    # indices_cache[m][r] = array of indices covered by (m, r)
    indices_cache = {}
    for m in divisors:
        indices_cache[m] = []
        for r in range(m):
            indices_cache[m].append(np.arange(r, N, m))
            
    # 3. Greedy Initialization
    counts = np.zeros(N, dtype=np.int32)
    residues = {}
    
    # To make greedy faster, we can just use r=0 initially and then improve?
    # Or do the greedy pass.
    # Greedy pass:
    # For each m, pick r that covers most uncovered.
    # This requires checking all r for each m.
    # Cost: sum(m * N/m) = N * len(D) = 52000 * 47 = 2.4M ops. Fast.
    
    for m in divisors:
        best_r = 0
        max_new = -1
        for r in range(m):
            idx = indices_cache[m][r]
            # Count how many are currently uncovered
            new_covered = np.sum(counts[idx] == 0)
            if new_covered > max_new:
                max_new = new_covered
                best_r = r
        residues[m] = best_r
        counts[indices_cache[m][best_r]] += 1
        
    uncovered = np.sum(counts == 0)
    best_uncovered = uncovered
    best_residues = residues.copy()
    
    # 4. Local Search
    random.seed(42)
    no_improve_steps = 0
    max_no_improve = 1000
    
    while time.time() - start_time < time_limit:
        # Pick random modulus
        m = random.choice(divisors)
        r_old = residues[m]
        
        best_r = r_old
        best_delta = 0
        
        # Evaluate all residues for m
        # We can optimize by only checking if it improves
        for r_new in range(m):
            if r_new == r_old:
                continue
                
            idx_old = indices_cache[m][r_old]
            idx_new = indices_cache[m][r_new]
            
            # L: elements in idx_old that have count 1 (will become 0)
            # G: elements in idx_new that have count 0 (will become 1)
            
            # Vectorized calculation
            # Note: idx_old and idx_new are arrays
            L = np.sum(counts[idx_old] == 1)
            G = np.sum(counts[idx_new] == 0)
            
            delta = L - G
            
            if delta < best_delta:
                best_delta = delta
                best_r = r_new
                
        if best_r != r_old:
            # Apply move
            idx_old = indices_cache[m][r_old]
            idx_new = indices_cache[m][best_r]
            
            counts[idx_old] -= 1
            counts[idx_new] += 1
            
            uncovered += best_delta
            
            residues[m] = best_r
            
            if uncovered < best_uncovered:
                best_uncovered = uncovered
                best_residues = residues.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1
            
        # If stuck, try a random move to escape local minimum?
        if no_improve_steps > max_no_improve:
            # Pick a random m and random r
            m_rand = random.choice(divisors)
            r_rand = random.randint(0, m_rand - 1)
            
            r_old_rand = residues[m_rand]
            if r_rand != r_old_rand:
                idx_old = indices_cache[m_rand][r_old_rand]
                idx_new = indices_cache[m_rand][r_rand]
                
                # Calculate delta for random move
                L = np.sum(counts[idx_old] == 1)
                G = np.sum(counts[idx_new] == 0)
                delta = L - G
                
                counts[idx_old] -= 1
                counts[idx_new] += 1
                uncovered += delta
                residues[m_rand] = r_rand
                
            no_improve_steps = 0

    # Return best found
    result = [[m, best_residues[m]] for m in divisors]
    return result
