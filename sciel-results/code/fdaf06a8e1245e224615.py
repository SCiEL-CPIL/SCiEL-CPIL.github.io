"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import sys

def construct():
    # Deadline guard: 150s limit, use 145s for safety
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
        
    primes = sieve(1000)
    n_primes = len(primes)
    
    # We want to maximize L such that every x in 1..L has coverage >= 2.
    # Let's aim for a target L and see if we can satisfy it.
    # Based on capacity analysis, L=1000 might be tight or impossible.
    # Let's try to construct for L=1000 first. If it fails, we fall back to lower L.
    
    # Strategy: Greedy Construction with Backtracking/Repair
    # 1. Initialize residues randomly or heuristically.
    # 2. Calculate coverage for 1..L_max (e.g., 1200 to be safe).
    # 3. Identify integers with coverage < 2.
    # 4. For each "weak" integer, find a prime that can cover it (by changing its residue) 
    #    that also covers another weak integer or doesn't hurt strong integers too much.
    
    # A more robust approach: Local Search (Hill Climbing) on the score L.
    # But the prompt asks for a DIFFERENT strategy.
    # Let's try "Greedy Set Cover" style:
    # Assign residues to primes one by one.
    # When assigning prime p, choose r_p to maximize the number of integers in [1, L_target] 
    # that would go from coverage 1 to 2, or 0 to 1, without dropping any from 2 to 1.
    
    # However, simple greedy might get stuck. 
    # Let's combine Greedy Init + Local Search.
    
    # First, let's determine a feasible L.
    # Calculate max possible coverage sum for L=1000.
    def max_capacity(L, primes):
        return sum(L // p for p in primes)
    
    # If capacity < 2*L, L is impossible.
    # Let's find the largest L such that capacity(L) >= 2*L.
    max_L_possible = 0
    for L in range(1, 2000):
        if max_capacity(L, primes) >= 2 * L:
            max_L_possible = L
        else:
            # Once it fails, it might fail forever? Not necessarily, but likely.
            # Actually, capacity grows linearly with L * sum(1/p).
            # If sum(1/p) < 2, then for large L, capacity < 2L.
            # sum(1/p) for p<=1000 is ~1.93. So 1.93 L < 2 L.
            # Thus, there is a hard limit on L!
            # We need to find the crossover point.
            pass
    
    # Let's find the exact limit L_max_cap
    L_limit = 0
    for L in range(1, 5000):
        if max_capacity(L, primes) >= 2 * L:
            L_limit = L
        else:
            break
            
    # L_limit is the theoretical maximum L based on total coverage slots.
    # We will try to achieve L = L_limit.
    
    # Greedy Construction
    # State: residues array, coverage array for 1..L_limit
    # We will use a priority queue or list of "deficits"
    
    # To speed up, we'll work with numpy if available, but standard list is fine for L~1000.
    # Let's use lists for simplicity and safety.
    
    target_L = L_limit
    if target_L == 0:
        return [0] * n_primes # Fallback, shouldn't happen
    
    # Initialize residues to 0
    residues = [0] * n_primes
    
    # Calculate initial coverage
    def calc_coverage(res, L):
        cov = [0] * (L + 1)
        for i, p in enumerate(primes):
            r = res[i]
            # x = r, r+p, r+2p ... <= L
            # If r == 0, x = p, 2p...
            # If r > 0, x = r, r+p...
            if r == 0:
                start = p
            else:
                start = r
            for x in range(start, L + 1, p):
                cov[x] += 1
        return cov
    
    # Greedy Step:
    # We want to fix the lowest coverage integers.
    # We'll iterate until no improvement or time limit.
    
    # Better Greedy: 
    # 1. Start with random residues.
    # 2. Compute coverage.
    # 3. Find x with min coverage.
    # 4. Try to change residue of a prime p that covers x (or could cover x) to improve min coverage.
    
    # Let's do a randomized restart + local search, but with a "Greedy" initialization 
    # that focuses on small primes first (they have high frequency).
    
    best_L = 0
    best_res = [0] * n_primes
    
    # Try to find a valid configuration for target_L
    # We'll use a simple local search on the "deficit" function.
    # Deficit(x) = max(0, 2 - coverage(x))
    # Total Deficit = sum(Deficit(x) for x in 1..L)
    # We want Total Deficit = 0.
    
    # If we can't reach 0 deficit for L_limit, we step down L.
    
    # Optimization: Use numpy for speed if L is large, but L is ~1000.
    # Python lists are okay.
    
    # Heuristic Init: 
    # Assign r_p such that it covers 1 and 2? 
    # If r_p = 1, covers 1, 1+p.
    # If r_p = 2, covers 2, 2+p.
    # Small primes should cover small integers densely.
    
    # Let's try a specific heuristic: 
    # For the first k primes, assign residues to cover 1..k twice.
    # This is complex.
    
    # Let's stick to: Random Init + Hill Climbing on Total Deficit for target L.
    # If stuck, reduce L.
    
    for current_L in range(target_L, 0, -10): # Step down by 10 to save time if stuck
        if time.time() > deadline - 5:
            break
            
        success = False
        best_res_current = None
        best_deficit = float('inf')
        
        # Try a few random restarts for this L
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            # Random Init
            res = [random.randint(0, p - 1) for p in primes]
            
            # Local Search
            cov = calc_coverage(res, current_L)
            
            # Calculate deficit
            deficit = 0
            for x in range(1, current_L + 1):
                if cov[x] < 2:
                    deficit += (2 - cov[x])
                    
            if deficit == 0:
                best_res_current = res
                success = True
                break
            
            best_deficit = min(best_deficit, deficit)
            if deficit < best_deficit:
                best_res_current = res
                best_deficit = deficit
                
            # Hill Climbing
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Pick a random prime to change
                idx = random.randint(0, n_primes - 1)
                p = primes[idx]
                old_r = res[idx]
                
                # Try a new residue
                # To be efficient, only try residues that cover a "weak" number
                # Weak numbers: x where cov[x] < 2
                weak_xs = [x for x in range(1, current_L + 1) if cov[x] < 2]
                if not weak_xs:
                    break
                
                # For each weak x, what residue r would cover it?
                # x = r mod p => r = x % p
                candidate_rs = set()
                for x in weak_xs[:10]: # Limit candidates
                    candidate_rs.add(x % p)
                
                if not candidate_rs:
                    continue
                    
                best_new_r = old_r
                best_new_deficit = deficit
                
                for r in candidate_rs:
                    if r == old_r:
                        continue
                    
                    # Simulate change
                    # Remove old coverage
                    old_cov = cov[:] # Copy for simulation? Expensive.
                    # Better: calculate delta
                    # Old r covers: old_r, old_r+p...
                    # New r covers: new_r, new_r+p...
                    
                    # Let's just recalculate coverage for this prime's contribution
                    # and update global cov.
                    
                    # Un-apply old
                    if old_r == 0:
                        start_old = p
                    else:
                        start_old = old_r
                    for x in range(start_old, current_L + 1, p):
                        cov[x] -= 1
                        
                    # Apply new
                    if r == 0:
                        start_new = p
                    else:
                        start_new = r
                    for x in range(start_new, current_L + 1, p):
                        cov[x] += 1
                        
                    # Calculate new deficit
                    new_deficit = 0
                    for x in range(1, current_L + 1):
                        if cov[x] < 2:
                            new_deficit += (2 - cov[x])
                            
                    if new_deficit < best_new_deficit:
                        best_new_deficit = new_deficit
                        best_new_r = r
                        
                    # Revert for next candidate
                    # Un-apply new
                    for x in range(start_new, current_L + 1, p):
                        cov[x] -= 1
                    # Re-apply old
                    for x in range(start_old, current_L + 1, p):
                        cov[x] += 1
                        
                # Apply best
                if best_new_r != old_r:
                    # Un-apply old
                    if old_r == 0:
                        start_old = p
                    else:
                        start_old = old_r
                    for x in range(start_old, current_L + 1, p):
                        cov[x] -= 1
                        
                    # Apply new
                    if best_new_r == 0:
                        start_new = p
                    else:
                        start_new = best_new_r
                    for x in range(start_new, current_L + 1, p):
                        cov[x] += 1
                        
                    res[idx] = best_new_r
                    deficit = best_new_deficit
                    improved = True
                    best_deficit = deficit
                    
                    if deficit == 0:
                        success = True
                        break
                        
        if success:
            best_res = best_res_current
            best_L = current_L
            break
        else:
            # If we didn't succeed, keep the best found so far if it's better
            if best_res_current and best_deficit < float('inf'):
                # Check if this res actually gives a valid L (maybe lower than current_L?)
                # We just track the best L we *could* prove.
                # For now, we just continue to lower L.
                pass
                
    # If we found a valid L, return the residues.
    # If best_L is 0, we return the heuristic from seed or random.
    # To ensure we return *something* valid, if we didn't find a full 2-coverage,
    # we should still return the best effort.
    # The score is the L where 2-coverage holds.
    
    # Final verification and cleanup
    if best_L == 0:
        # Fallback to seed heuristic
        return [((i + 2) // 2) % p for i, p in enumerate(primes)]
        
    return best_res

# EVOLVE-BLOCK-END
