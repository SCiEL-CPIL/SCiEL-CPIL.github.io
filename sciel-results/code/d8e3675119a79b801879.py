import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Primes
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = [False] * len(sieve[i*i::i])
        return [i for i, is_p in enumerate(sieve) if is_p]
        
    primes = get_primes(1000)
    n_primes = len(primes)
    
    L_max = 2000 # Search horizon
    
    # 2. Precompute positions
    # pos[i] is a list of numpy arrays, pos[i][r] contains indices in [1, L_max]
    pos = []
    for p in primes:
        p_list = []
        for r in range(p):
            if r == 0:
                arr = np.arange(p, L_max + 1, p)
            else:
                arr = np.arange(r, L_max + 1, p)
            p_list.append(arr)
        pos.append(p_list)
        
    # 3. Initialization
    # Greedy: maximize coverage in [1, L_max]
    residues = [0] * n_primes
    coverage = np.zeros(L_max + 1, dtype=np.int8)
    
    # Sort primes by size? Small primes cover more points.
    # Let's process small primes first to build a dense core.
    order = list(range(n_primes))
    
    for i in order:
        p = primes[i]
        best_r = 0
        best_gain = -1
        for r in range(p):
            # Gain is number of positions where coverage increases from 0->1 or 1->2
            # We want to maximize sum( min(1, max(0, coverage[x]+1-1)) )? 
            # Simpler: count how many x in pos[i][r] have coverage[x] < 2.
            gain = np.sum(coverage[pos[i][r]] < 2)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        residues[i] = best_r
        coverage[pos[i][best_r]] += 1
        
    def find_L(coverage):
        L = 0
        # Find first index where coverage < 2
        # coverage[0] is not used, we care about 1..L_max
        for x in range(1, L_max + 1):
            if coverage[x] >= 2:
                L = x
            else:
                break
        return L

    current_L = find_L(coverage)
    best_L = current_L
    best_residues = residues[:]
    
    # 4. Optimization: Repair + SA
    rng = random.Random(42)
    
    # Helper: Calculate deficit
    # deficit = sum( max(0, 2 - coverage[x]) ) for x in 1..L_max
    # We track this incrementally.
    
    # Initialize deficit
    deficit = np.sum(np.maximum(0, 2 - coverage[1:L_max+1]))
    
    T = 1.0
    alpha = 0.999
    
    while time.time() < deadline:
        if T < 0.01:
            break
            
        # --- Repair Phase ---
        # Try to extend L by fixing L+1
        if current_L < L_max:
            target = current_L + 1
            if coverage[target] < 2:
                # Find a prime that doesn't cover target
                # i.e., residues[i] != target % primes[i]
                candidates = []
                for i in range(n_primes):
                    p = primes[i]
                    if residues[i] != target % p:
                        candidates.append(i)
                
                if candidates:
                    # Pick a random candidate or the one with minimal cost
                    # Cost: how many x in [1, L] will lose coverage?
                    # We only care if they drop below 2.
                    best_i = -1
                    best_cost = float('inf')
                    for i in candidates:
                        p = primes[i]
                        old_r = residues[i]
                        new_r = target % p
                        old_pos = pos[i][old_r]
                        new_pos = pos[i][new_r]
                        
                        # Calculate cost: loss of coverage in [1, L]
                        # x loses coverage if x in old_pos and x not in new_pos
                        # and coverage[x] becomes 1 (was 2)
                        
                        # Efficient check:
                        # Intersect old_pos with [1, L]
                        # Check which of those are NOT in new_pos
                        # And have coverage == 2
                        
                        # Using numpy:
                        # mask = (old_pos <= L) & (coverage[old_pos] == 2)
                        # bad = mask & ~np.isin(old_pos, new_pos)
                        # cost = sum(bad)
                        
                        # np.isin is slow. Let's use a set for new_pos if small?
                        # Or just iterate if L is small? L is ~400.
                        # Let's just compute delta deficit for the whole range?
                        # No, let's just check if it breaks [1, L].
                        
                        # Simpler: just try the move, check if find_L decreases.
                        # If it does, revert.
                        # This is O(L) per check.
                        
                        pass # We will do a "try and revert" approach for repair to keep code simple and robust.
                    
                    # Try one random candidate from repair
                    i = rng.choice(candidates)
                    p = primes[i]
                    old_r = residues[i]
                    new_r = target % p
                    
                    old_pos = pos[i][old_r]
                    new_pos = pos[i][new_r]
                    
                    # Apply
                    coverage[old_pos] -= 1
                    coverage[new_pos] += 1
                    residues[i] = new_r
                    
                    new_L = find_L(coverage)
                    
                    if new_L >= current_L:
                        # Accept
                        current_L = new_L
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = residues[:]
                        # Update deficit
                        # Recalculate deficit? Or update incrementally?
                        # Incremental is better.
                        # delta_deficit = sum( max(0, 2-c1) - max(0, 2-c0) ) for affected x
                        # We already applied, so we need to revert deficit logic or just recalc if needed.
                        # For simplicity in repair, we don't strictly need deficit if we use find_L.
                        # But SA needs a cost function.
                        # Let's maintain deficit.
                        
                        # Actually, let's just recalculate deficit periodically or on every move?
                        # Recalc is O(L_max). 2000 ops. Very fast.
                        deficit = np.sum(np.maximum(0, 2 - coverage[1:L_max+1]))
                    else:
                        # Revert
                        coverage[old_pos] += 1
                        coverage[new_pos] -= 1
                        residues[i] = old_r
                        # Deficit unchanged
                        pass
        
        # --- SA Phase ---
        # Pick random prime
        i = rng.randrange(n_primes)
        p = primes[i]
        old_r = residues[i]
        new_r = rng.randrange(p)
        
        if old_r == new_r:
            continue
            
        old_pos = pos[i][old_r]
        new_pos = pos[i][new_r]
        
        # Calculate delta deficit
        # Affected positions are union of old and new
        # We can just compute new deficit and old deficit?
        # No, incremental.
        
        # Let's compute delta by iterating over affected positions?
        # Union size is approx L_max/p.
        # For p=2, size is 1000.
        # For p=1000, size is 2.
        # Average is small.
        
        # Create a set of affected indices for O(1) lookup?
        # Or just use numpy boolean masks?
        
        # Simpler:
        # old_deficit_part = sum(max(0, 2-coverage[x]) for x in old_pos)
        # This is hard to do incrementally without storing per-position deficit.
        
        # Let's store `deficit_contrib` array?
        # No, let's just compute the change in total deficit.
        
        # Total deficit D = sum_x d(x)
        # D' = D - sum_{x in old} d(x) + sum_{x in old} d(x-1) ... wait.
        # c_new(x) = c(x) - 1 if x in old, +1 if x in new.
        
        # Let's just compute the new deficit for the affected positions.
        # affected = np.unique(np.concatenate([old_pos, new_pos]))
        # This is fast enough.
        
        affected = np.unique(np.concatenate([old_pos, new_pos]))
        
        old_def_val = 0
        new_def_val = 0
        for x in affected:
            c0 = coverage[x]
            c1 = c0
            if x in old_pos: # This check is O(N) in python. Bad.
                c1 -= 1
            if x in new_pos:
                c1 += 1
            old_def_val += max(0, 2 - c0)
            new_def_val += max(0, 2 - c1)
            
        delta_def = new_def_val - old_def_val
        
        accept = False
        if delta_def < 0:
            accept = True
        elif delta_def == 0:
            # Accept neutral moves to explore
            if rng.random() < 0.5:
                accept = True
        else:
            if delta_def < T * 10:
                if rng.random() < np.exp(-delta_def / T):
                    accept = True
                    
        if accept:
            coverage[old_pos] -= 1
            coverage[new_pos] += 1
            residues[i] = new_r
            deficit += delta_def
            
            # Check L
            new_L = find_L(coverage)
            if new_L > current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
            # If L decreased, we might want to revert? 
            # SA allows worse moves. But we track best_L.
            
        T *= alpha
        
        # Periodic polish?
        # Every 1000 steps, do a coordinate descent?
        
    # Final Polish: Coordinate Descent
    # Iterate primes, for each, try all residues, keep best L.
    # This is expensive but good for final squeeze.
    # We have ~10s left.
    
    # Use best_residues as base
    residues = best_residues[:]
    coverage = np.zeros(L_max + 1, dtype=np.int8)
    for i in range(n_primes):
        coverage[pos[i][residues[i]]] += 1
        
    current_L = find_L(coverage)
    
    # Simple local search on best_residues
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(n_primes):
            p = primes[i]
            old_r = residues[i]
            best_local_L = current_L
            best_local_r = old_r
            best_local_cov = coverage.copy() # Save state? No, just revert.
            
            # Try all r
            for r in range(p):
                if r == old_r: continue
                # Apply
                coverage[pos[i][old_r]] -= 1
                coverage[pos[i][r]] += 1
                new_L = find_L(coverage)
                if new_L > best_local_L:
                    best_local_L = new_L
                    best_local_r = r
                # Revert
                coverage[pos[i][r]] -= 1
                coverage[pos[i][old_r]] += 1
                
            if best_local_r != old_r:
                # Apply best
                coverage[pos[i][old_r]] -= 1
                coverage[pos[i][best_local_r]] += 1
                residues[i] = best_local_r
                current_L = best_local_L
                improved = True
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
                    
    return [int(r) for r in best_residues]
