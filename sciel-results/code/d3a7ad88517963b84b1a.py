import random
import time

def construct():
    # Generate primes up to 1000
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(p**0.5) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
    
    n_primes = len(primes)  # 168
    MAX_X = 5000  # We aim for L >= 1000, but allow up to 5000 for search
    
    def solve_repair(init_residues=None):
        if init_residues is None:
            # Start with 0s, which is a structured baseline
            residues = [0] * n_primes
        else:
            residues = init_residues[:]
            
        cov = [0] * (MAX_X + 1)
        
        # Initial coverage
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            x = start
            while x <= MAX_X:
                cov[x] += 1
                x += p
        
        L = 0
        for x in range(1, MAX_X + 1):
            if cov[x] < 2:
                break
            L = x
            
        # Identify gaps
        # We maintain a set/list of x where cov[x] < 2
        # To be efficient, we just scan from 1 to L+100 or so each time?
        # No, we can maintain a list of "broken" spots.
        
        # For simplicity in this step, we just loop x from 1 to MAX_X
        # and try to fix the first one found that is < 2.
        # This is O(MAX_X * N_primes * MAX_X/p) which might be slow.
        
        # Optimized:
        # Keep track of L.
        # We try to extend L or fix holes in [1, L].
        
        max_iter = 10000
        iter_count = 0
        
        while iter_count < max_iter:
            iter_count += 1
            
            # Find the first x in [1, L+100] where cov[x] < 2
            # We prioritize x <= L (holes) then x = L+1 (extension)
            target_x = -1
            # Check for holes in [1, L]
            for x in range(1, L + 1):
                if cov[x] < 2:
                    target_x = x
                    break
            
            # If no holes, try to extend
            if target_x == -1:
                if L < MAX_X and cov[L+1] < 2:
                    target_x = L + 1
                else:
                    # Check further ahead?
                    # If L+1 is covered, L might extend.
                    # Let's just recompute L.
                    new_L = 0
                    for x in range(1, MAX_X + 1):
                        if cov[x] < 2:
                            break
                        new_L = x
                    if new_L > L:
                        L = new_L
                        continue
                    else:
                        # Stuck?
                        # If we can't extend and no holes, we are at a local max for this config.
                        # We can try a random perturbation.
                        break
            
            # We have a target_x to fix.
            # Find best prime to change.
            best_p_idx = -1
            best_cost = float('inf')
            best_new_r = -1
            best_gain = -1 # How many other gaps this change fixes
            
            # Candidates: primes p where residues[i] != target_x % p
            for i, p in enumerate(primes):
                cur_r = residues[i]
                new_r = target_x % p
                if new_r == cur_r:
                    continue
                
                # Calculate cost: damage to [1, L]
                cost = 0
                if cur_r == 0:
                    start_y = p
                else:
                    start_y = cur_r
                
                y = start_y
                while y <= L:
                    if cov[y] == 2:
                        cost += 1
                        if cost >= best_cost:
                            break
                    y += p
                
                # Calculate gain: how many other x' in [target_x, L+100] with cov[x']<2 are covered by new_r?
                # This is expensive. Let's approximate or just use cost.
                # Actually, if we fix target_x, we might fix others if they share the same prime/residue.
                
                if cost < best_cost:
                    best_cost = cost
                    best_p_idx = i
                    best_new_r = new_r
                    best_gain = 0 # Placeholder
                    
            if best_p_idx == -1:
                # No prime can cover target_x without being already there?
                # If all primes cover target_x, cov[target_x] >= 168.
                # But cov[target_x] < 2. Contradiction.
                # So this shouldn't happen unless target_x is not reachable?
                # Wait, if target_x < 2, no prime covers it?
                # Primes are >= 2.
                # If target_x = 1, r_p = 1 covers it.
                # So there is always a candidate.
                break
                
            # Apply change
            p = primes[best_p_idx]
            old_r = residues[best_p_idx]
            new_r = best_new_r
            
            # Update cov
            if old_r == 0:
                start = p
            else:
                start = old_r
            y = start
            while y <= MAX_X:
                cov[y] -= 1
                y += p
                
            if new_r == 0:
                start = p
            else:
                start = new_r
            y = start
            while y <= MAX_X:
                cov[y] += 1
                y += p
                
            residues[best_p_idx] = new_r
            
            # Update L
            # L might decrease if we broke a spot.
            # L might increase if we fixed a spot.
            # Recalculate L from scratch is O(MAX_X).
            # MAX_X is 5000. 5000 * 10000 iterations = 50M ops. Might be tight in 120s.
            # Let's optimize L update.
            
            # Check if L extended
            # If we fixed target_x and target_x == L+1, check L+2, etc.
            # If we broke a spot y < L, L drops to y-1.
            
            # Simple approach: Recalculate L.
            new_L = 0
            for x in range(1, MAX_X + 1):
                if cov[x] < 2:
                    break
                new_L = x
            
            if new_L < L:
                # We broke the prefix.
                # This is allowed in repair, but we hope to fix it later.
                L = new_L
            else:
                L = new_L
                
            # If we are stuck in a loop (L doesn't improve for many steps), we might need to restart.
            # For now, just continue.
            
        return residues, L

    best_L = 0
    best_res = None
    start_time = time.time()
    time_budget = 115
    
    # Run multiple strategies
    # 1. Start with 0s
    # 2. Start with random
    # 3. Start with r_p = 1 for all p (covers 1, p+1, ...)
    
    inits = [
        [0] * n_primes,
        [1] * n_primes, # Careful: 1 < p for all p? No, p=2, r=1 is ok. p=3, r=1 ok.
        [random.randint(0, p - 1) for p in primes]
    ]
    
    for init in inits:
        if time.time() - start_time > time_budget:
            break
        res, L = solve_repair(init)
        if L > best_L:
            best_L = L
            best_res = res[:]
            
    # If we still have time, do random restarts
    while time.time() - start_time < time_budget:
        init = [random.randint(0, p - 1) for p in primes]
        res, L = solve_repair(init)
        if L > best_L:
            best_L = L
            best_res = res[:]
            
    if best_res is None:
        return [0] * 168
        
    return best_res
