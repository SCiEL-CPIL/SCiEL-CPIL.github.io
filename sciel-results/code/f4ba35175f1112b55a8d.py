"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(1000)
    n = len(primes)  # 168 primes

    # Precompute prime indices for fast access
    primes_arr = np.array(primes)

    def evaluate(residues, L_max=100000):
        """Return the largest L such that every integer 1..L satisfies at least 2 congruences."""
        residues = np.array(residues)
        
        # We'll check incrementally. 
        # For efficiency, we'll compute coverage for a range of x values
        coverage = np.zeros(L_max + 1, dtype=np.int32)
        for i in range(n):
            p = primes_arr[i]
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start > L_max:
                continue
            # x = start, start + p, start + 2p, ... <= L_max
            idx = np.arange(start, L_max + 1, p)
            coverage[idx] += 1
        
        # Find the largest L such that coverage[1..L] >= 2
        valid = coverage[1:] >= 2
        if not valid[0]:
            return 0
        false_positions = np.where(~valid)[0]
        if len(false_positions) == 0:
            return L_max
        return false_positions[0]

    def seed_solution():
        return [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    best_residues = seed_solution()
    best_score = evaluate(best_residues)

    # Greedy Sequential Constraint Repair
    # Strategy:
    # 1. Start with a valid small prefix (e.g., L=1 or 2).
    # 2. Try to extend the prefix to L+1.
    # 3. If L+1 is not covered by >= 2 congruences, find a prime p_i that covers L (or L-1) 
    #    and reassign its residue to cover L+1.
    # 4. To avoid breaking previous coverage, we prefer primes that cover "many" previous points 
    #    or have large modulus (more flexibility).
    # 5. We also perform local "swap" repairs if stuck.

    def greedy_repair(residues, max_L=10000):
        """
        Attempt to extend the doubly-covered prefix as far as possible using greedy repair.
        """
        residues = list(residues)
        # We maintain the current valid prefix length
        # First, determine the current valid prefix
        current_L = evaluate(residues, min(max_L, 2000)) # quick check
        
        if current_L == 0:
            # Try to fix L=1
            # Need 2 primes to cover 1.
            # Pick 2 smallest primes? No, pick 2 largest for flexibility?
            # Actually, for L=1, any 2 primes can cover 1 if residues are 1%p.
            # Let's just set residues for first 2 primes to 1%p.
            if n >= 2:
                residues[0] = 1 % primes[0]
                residues[1] = 1 % primes[1]
                current_L = evaluate(residues, min(max_L, 2000))
                if current_L == 0:
                    return residues, 0
            else:
                return residues, 0
        
        # Now try to extend from current_L + 1
        # We need to be careful: evaluate is expensive. 
        # Instead of calling evaluate every step, we maintain a coverage array 
        # for a window around current_L.
        
        # Let's use a sliding window coverage approach.
        # Window size: maybe 1000?
        # We'll track coverage for x in [current_L - 500, current_L + 500]
        
        # Actually, simpler: just call evaluate periodically or use a fast local check.
        # Since max_L is likely < 10000, we can afford to recompute coverage 
        # for the relevant range.
        
        # Let's define a helper to get coverage for a specific range
        def get_coverage_range(residues, start, end):
            """Returns coverage array for x in [start, end]"""
            cov = np.zeros(end - start + 1, dtype=np.int32)
            for i in range(n):
                p = primes[i]
                r = residues[i]
                # Find first x in [start, end] such that x % p == r
                # x = r + k*p
                # k_min = ceil((start - r) / p)
                # k_max = floor((end - r) / p)
                if r == 0:
                    # x = p, 2p, ...
                    k_min = max(1, math.ceil(start / p))
                else:
                    k_min = max(0, math.ceil((start - r) / p))
                
                k_max = math.floor((end - r) / p)
                
                if k_min > k_max:
                    continue
                
                # Indices in cov array: x - start
                # x = r + k*p
                # First x: r + k_min*p
                first_x = r + k_min * p
                # Last x: r + k_max*p
                # Step: p
                # Indices: first_x - start, (first_x + p) - start, ...
                idxs = np.arange(first_x - start, end - start + 1, p)
                cov[idxs] += 1
            return cov

        # Start from current_L
        # We want to ensure 1..current_L are covered.
        # Let's verify this assumption.
        # If not, we might need to backtrack, but greedy repair assumes we are at a valid state.
        
        # We will iterate target_x from current_L + 1 to max_L
        # But we need to ensure that changing residues doesn't break 1..current_L.
        # This is the hard part.
        
        # Alternative: 
        # We maintain the full coverage array for 1..max_L? 
        # max_L=10000, n=168. 
        # Building coverage array takes ~ n * (L/p) operations.
        # Sum(L/p) ~ L * sum(1/p) ~ L * log(log(1000)) ~ 10000 * 2 = 20000.
        # Very fast. 
        # So we can just rebuild the coverage array for 1..max_L every time we change a residue?
        # No, that's too slow if we do it for every x.
        
        # Better: 
        # Maintain coverage array for 1..max_L.
        # When we change residue for prime i, we subtract the old contribution and add the new one.
        
        max_eval_L = 5000 # Let's aim for 5000 first, extend if time permits
        coverage = np.zeros(max_eval_L + 1, dtype=np.int32)
        
        def update_coverage(residues):
            nonlocal coverage
            coverage[:] = 0
            for i in range(n):
                p = primes[i]
                r = residues[i]
                if r == 0:
                    start = p
                else:
                    start = r
                if start > max_eval_L:
                    continue
                idx = np.arange(start, max_eval_L + 1, p)
                coverage[idx] += 1
        
        update_coverage(residues)
        
        # Find current valid prefix
        valid = coverage[1:max_eval_L+1] >= 2
        if not valid[0]:
            # Fix L=1
            # Find 2 primes that can cover 1
            # Just pick 2 random primes and set residues to 1%p
            idxs = random.sample(range(n), 2)
            for i in idxs:
                residues[i] = 1 % primes[i]
            update_coverage(residues)
            valid = coverage[1:max_eval_L+1] >= 2
            if not valid[0]:
                return residues, 0
        
        current_L = 0
        for i in range(max_eval_L):
            if valid[i]:
                current_L = i + 1
            else:
                break
        
        # Now extend
        for target_x in range(current_L + 1, max_eval_L + 1):
            if time.time() > deadline - 5:
                break
                
            if coverage[target_x] >= 2:
                # Already covered, just move on
                current_L = target_x
                continue
            
            # Need to add coverage to target_x
            # Find primes that currently cover target_x
            # And primes that do NOT cover target_x
            # We want to change a prime that does NOT cover target_x to cover it.
            # But we must ensure we don't break coverage for 1..current_L.
            
            # Candidate primes:
            # 1. Primes that cover target_x - 1? No, they might be needed there.
            # 2. Primes that are "redundant" in the prefix?
            
            # Strategy: 
            # For each prime i, if we change r_i to target_x % p_i, 
            # how many points in 1..current_L lose coverage?
            # This is expensive to check for all i.
            
            # Heuristic: 
            # Try primes with large modulus first (more "space" to move without affecting many points in the small prefix? No, larger modulus means fewer points in prefix).
            # Actually, smaller modulus primes hit more points in the prefix. 
            # So changing a large modulus prime is safer for the prefix.
            
            # Let's sort primes by modulus descending.
            # For each prime i in sorted order:
            #   new_r = target_x % primes[i]
            #   if new_r == residues[i]: continue
            #   Check if changing r_i to new_r breaks any point in 1..current_L.
            #   To check this, we need to know the coverage of points that ONLY this prime covers.
            #   This requires maintaining "exclusive coverage" or similar.
            
            # Simpler Heuristic: 
            # Just try changing a random prime that doesn't cover target_x.
            # Then verify the prefix.
            
            # Let's collect primes that do NOT cover target_x
            non_covering = []
            for i in range(n):
                if (target_x - residues[i]) % primes[i] != 0:
                    non_covering.append(i)
            
            if not non_covering:
                # All primes cover target_x? Impossible if coverage < 2.
                # Actually, if coverage < 2, then at most 1 prime covers it.
                # So non_covering has size >= n-1.
                pass
            
            # Shuffle non_covering, but prioritize larger primes
            non_covering.sort(key=lambda i: -primes[i])
            
            improved = False
            for i in non_covering:
                old_r = residues[i]
                new_r = target_x % primes[i]
                if new_r == old_r:
                    continue
                
                residues[i] = new_r
                # Update coverage
                # Remove old contribution
                p = primes[i]
                if old_r == 0:
                    old_start = p
                else:
                    old_start = old_r
                if old_start <= max_eval_L:
                    idx_old = np.arange(old_start, max_eval_L + 1, p)
                    coverage[idx_old] -= 1
                
                # Add new contribution
                if new_r == 0:
                    new_start = p
                else:
                    new_start = new_r
                if new_start <= max_eval_L:
                    idx_new = np.arange(new_start, max_eval_L + 1, p)
                    coverage[idx_new] += 1
                
                # Check if prefix 1..current_L is still valid
                # And if target_x is now covered
                if coverage[target_x] >= 2:
                    # Check prefix
                    # We only need to check points that might have been affected
                    # Affected points are those where coverage dropped from 2 to 1 or 1 to 0.
                    # Since we only changed one prime, the points affected are 
                    # those in the symmetric difference of the old and new arithmetic progressions.
                    # This is complex. 
                    # For now, just check the whole prefix? 
                    # current_L is small initially. 
                    # If current_L is large, this is slow.
                    
                    # Optimization: 
                    # We know that coverage[target_x] increased by 1 (or stayed same if it was already covered, but we filtered those).
                    # Actually, if it was 0 or 1, it is now 1 or 2.
                    # If it was 1, it is now 2. Good.
                    # If it was 0, it is now 1. Bad, need another prime.
                    
                    # Let's check if any point in 1..current_L has coverage < 2.
                    # We can do this by checking the minimum coverage in 1..current_L.
                    # min_cov = np.min(coverage[1:current_L+1])
                    # if min_cov >= 2:
                    #     current_L = target_x
                    #     improved = True
                    #     break
                    # else:
                    #     # Revert
                    #     residues[i] = old_r
                    #     # Restore coverage
                    #     if new_start <= max_eval_L:
                    #         idx_new = np.arange(new_start, max_eval_L + 1, p)
                    #         coverage[idx_new] -= 1
                    #     if old_start <= max_eval_L:
                    #         idx_old = np.arange(old_start, max_eval_L + 1, p)
                    #         coverage[idx_old] += 1
                    
                    # This check is O(current_L). 
                    # If we do this for many i, it's O(n * current_L).
                    # With n=168 and current_L=1000, that's 168,000 ops. Fast.
                    
                    if np.min(coverage[1:current_L+1]) >= 2:
                        current_L = target_x
                        improved = True
                        break
                    else:
                        # Revert
                        residues[i] = old_r
                        if new_start <= max_eval_L:
                            idx_new = np.arange(new_start, max_eval_L + 1, p)
                            coverage[idx_new] -= 1
                        if old_start <= max_eval_L:
                            idx_old = np.arange(old_start, max_eval_L + 1, p)
                            coverage[idx_old] += 1
            
            if not improved:
                # Stuck? Try a more aggressive repair:
                # Maybe we need to change TWO primes?
                # Or accept that we can't extend further with this greedy approach.
                # Let's try a few random restarts of the greedy process from current state.
                # Or just break.
                break
        
        return residues, current_L

    # Run greedy repair
    random.seed(42)
    
    # Try a few different initial seeds?
    # The seed_solution is fixed. 
    # But we can randomize the order of prime processing in greedy_repair?
    # The greedy_repair uses sorted primes by size.
    
    # Let's just run it once.
    repaired_residues, repaired_score = greedy_repair(best_residues, max_L=5000)
    
    if repaired_score > best_score:
        best_score = repaired_score
        best_residues = repaired_residues
        
    # If we have time, try to extend further with a different strategy?
    # Or just return.
    
    # Final evaluation to be sure
    final_score = evaluate(best_residues)
    if final_score > best_score:
        best_score = final_score
        
    return best_residues
# EVOLVE-BLOCK-END
